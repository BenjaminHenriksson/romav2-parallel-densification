"""CPU regression tests; no weights, downloads, or GPU inference required."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("driver", ROOT / "densify_parallel_gubbangen.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
driver.install_plugin_imports()

from densiplug.core import matcher, pipeline
from densiplug.core.camera_models import CameraRecord
from densiplug.core.config import DensePipelineConfig


class FakeRoMa:
    """Expose the real preset sizes without loading network weights."""
    class Cfg:
        def __init__(self, **kwargs):
            pass

    def __init__(self, cfg):
        self.apply_setting("precise")

    def apply_setting(self, setting):
        low, high = {"turbo": (320, None), "fast": (512, None),
                     "base": (640, None), "precise": (800, 1280)}[setting]
        self.W_lr = self.H_lr = low
        self.W_hr = self.H_hr = high
        self.bidirectional = high is not None

    def to(self, device):
        return self

    def eval(self):
        return self

    def _load_image(self, image):
        return torch.from_numpy(np.array(image)).permute(2, 0, 1)[None].float() / 255

    def f(self, image):
        self.low_input = image.clone()
        return [image]

    def match_from_features(self, *, f_list_A, img_A_lr, imB, img_A_hr):
        self.high_input = img_A_hr.clone()
        self.neighbor_size = imB.size
        h, w = img_A_hr.shape[-2:]
        return {"warp_AB": torch.zeros(1, h, w, 2),
                "overlap_AB": torch.ones(1, h, w, 1)}


def camera(uid, image, mask, width, height, tx=0):
    K = np.array([[100., 0, width / 2], [0, 100., height / 2], [0, 0, 1]])
    R = np.eye(3)
    t = np.array([tx, 0., 0.])
    return CameraRecord(uid=uid, image_path=str(image), mask_path=str(mask) if mask else None,
                        width=width, height=height, K=K, R=R, t=t,
                        P=K @ np.column_stack([R, t]), C=-t)


class ResolutionTests(unittest.TestCase):
    def test_presets_load_final_resolution_and_skip_unused_reverse_branch(self):
        for mode, expected in [("turbo", 320), ("fast", 512), ("base", 640),
                               ("high", 960), ("precise", 1280)]:
            with self.subTest(mode=mode), patch.object(matcher, "RoMaV2", FakeRoMa):
                m = matcher.RomaMatcher(device="cpu", setting=mode)
                self.assertEqual((m.w_resized, m.h_resized), (expected, expected))
                self.assertFalse(m.model.bidirectional)
                m.close()

    def test_precise_preserves_source_detail_and_masks_for_both_images(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(matcher, "RoMaV2", FakeRoMa):
            p = Path(tmp)
            # Alternating pixels are lost by a downsample-to-800/upscale round trip.
            rgb = np.repeat(((np.indices((1280, 1280)).sum(0) % 2) * 255)
                            .astype(np.uint8)[..., None], 3, axis=2)
            mask = np.full((1280, 1280), 255, np.uint8)
            mask[:, :128] = 0
            Image.fromarray(rgb).save(p / "image.png")
            Image.fromarray(mask).save(p / "mask.png")
            records = [camera(i, p / "image.png", p / "mask.png", 1280, 1280, -i)
                       for i in range(2)]
            m = matcher.RomaMatcher(device="cpu", setting="precise")
            ctx = pipeline._PackContext(pipeline._build_camera_lookup(records),
                                        np.array([[1], [0]]), 1, m.w_resized, m.h_resized)
            packed = pipeline._pack_reference_batch(0, ctx)
            expected = rgb.copy()
            expected[:, :128] = 0
            np.testing.assert_array_equal(packed.imA_np, expected)
            np.testing.assert_array_equal(packed.nn_arrays[0], expected)
            np.testing.assert_array_equal(packed.maskA_np, mask > 0)
            np.testing.assert_array_equal(packed.nn_masks[0], mask > 0)
            result = m.match_grids(Image.fromarray(packed.imA_np), Image.fromarray(packed.nn_arrays[0]))
            self.assertEqual(tuple(m.model.low_input.shape[-2:]), (800, 800))
            self.assertEqual(tuple(m.model.high_input.shape[-2:]), (1280, 1280))
            np.testing.assert_array_equal(m.model.high_input[0].permute(1, 2, 0).numpy() * 255, expected)
            self.assertEqual(m.model.neighbor_size, (1280, 1280))
            self.assertEqual(tuple(result[0].shape), (1280, 1280, 4))
            m.close()

    def test_normalized_matches_preserve_geometry_across_processing_sizes(self):
        records = [camera(0, "unused", None, 200, 100), camera(1, "unused", None, 200, 100, -1)]
        lookup = pipeline._build_camera_lookup(records)
        ys, xs = np.meshgrid(np.linspace(30., 60., 4), np.linspace(80., 110., 4), indexing="ij")
        warp = np.stack([xs / 100 - 1, ys / 50 - 1, (xs - 20) / 100 - 1, ys / 50 - 1], -1)
        expected = np.column_stack([(xs.ravel() - 100) / 20, (ys.ravel() - 50) / 20, np.full(16, 5)])
        for size in [512, 640, 960, 1280]:
            with self.subTest(size=size):
                rgb = np.full((size, size, 3), 128, np.uint8)
                packed = pipeline._PackedReferenceBatch(0, "unused", rgb, None, 200, 100,
                                                         [1], [None], [rgb])
                matched = pipeline._MatchedReference(packed, [torch.tensor(warp, dtype=torch.float32)],
                                                      [torch.ones(4, 4)], {1: 1}, {1: rgb})
                config = DensePipelineConfig(output_path="unused", no_filter=True, matches_per_ref=16)
                ctx = pipeline._TriangulationContext(lookup, config, .9, size, size)
                result = pipeline._triangulate_ref(matched, ctx)
                self.assertIsNotNone(result)
                # SVD introduces tiny row-wise Y differences: compare the point
                # sets without letting those roundoff errors reorder each row.
                actual = result.xyz[np.lexsort((np.round(result.xyz[:, 0], 5),
                                                np.round(result.xyz[:, 1], 5)))]
                order = np.lexsort((expected[:, 0], expected[:, 1]))
                np.testing.assert_allclose(actual, expected[order], atol=2e-5)
                np.testing.assert_allclose(result.rgb, 128 / 255, atol=1e-6)
                self.assertLess(float(result.err.max()), 1e-4)


if __name__ == "__main__":
    unittest.main()
