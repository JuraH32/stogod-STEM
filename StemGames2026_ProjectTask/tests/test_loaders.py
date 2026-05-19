from __future__ import annotations

import unittest
from pathlib import Path

from StemGames2026_ProjectTask.pointcloud import load_project_scenes, load_scene

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_IMAGES_ROOT = PROJECT_ROOT / "TestImages"


class LoaderTests(unittest.TestCase):
    def test_box_scene_loads_known_camera_views(self) -> None:
        dataset = load_scene(TEST_IMAGES_ROOT / "Box")

        self.assertEqual(dataset.scene_name, "Box")
        self.assertEqual(len(dataset.views), 12)
        self.assertFalse(dataset.requires_pose_estimation)
        self.assertAlmostEqual(dataset.views[0].intrinsics.fx, 960.0, places=6)
        self.assertAlmostEqual(dataset.views[0].intrinsics.cx, 960.0, places=6)
        self.assertAlmostEqual(dataset.views[0].intrinsics.cy, 540.0, places=6)
        self.assertEqual(dataset.views[0].pose_status, "known")
        self.assertIsNotNone(dataset.views[0].pose)
        self.assertAlmostEqual(dataset.views[0].pose.position[0], -588.0, places=6)

    def test_box_loader_handles_metadata_with_missing_colon(self) -> None:
        dataset = load_scene(TEST_IMAGES_ROOT / "Box")
        fifth_view = dataset.views[4]

        self.assertEqual(fifth_view.index, 5)
        self.assertIsNotNone(fifth_view.pose)
        self.assertAlmostEqual(fifth_view.pose.right[0], 0.829, places=6)
        self.assertAlmostEqual(fifth_view.pose.right[1], 0.559, places=6)
        self.assertAlmostEqual(fifth_view.pose.up[2], 0.866, places=6)

    def test_fountain_scene_loads_without_poses(self) -> None:
        dataset = load_scene(TEST_IMAGES_ROOT / "Fountain")
        first_view = dataset.views[0]

        self.assertEqual(dataset.scene_name, "Fountain")
        self.assertEqual(len(dataset.views), 11)
        self.assertTrue(dataset.requires_pose_estimation)
        self.assertIsNone(first_view.pose)
        self.assertEqual(first_view.pose_status, "needs_estimation")
        self.assertAlmostEqual(first_view.intrinsics.fx, 2759.48, places=2)
        self.assertAlmostEqual(first_view.intrinsics.cx, 1520.69, places=2)
        self.assertEqual(first_view.intrinsics.image_size, (3072, 2048))
        self.assertAlmostEqual(first_view.intrinsics.fov_degrees, 58.20, places=2)
        self.assertEqual(first_view.metadata["reported_horizontal_fov_degrees"], 84.0)
        self.assertEqual(first_view.metadata["reported_fov_matches_intrinsics"], 0)

    def test_statue_scene_preserves_matching_fov_hint(self) -> None:
        dataset = load_scene(TEST_IMAGES_ROOT / "Statue")
        first_view = dataset.views[0]

        self.assertAlmostEqual(first_view.intrinsics.fov_degrees, 90.0, places=6)
        self.assertEqual(first_view.metadata["reported_horizontal_fov_degrees"], 90.0)
        self.assertEqual(first_view.metadata["reported_fov_matches_intrinsics"], 1)

    def test_project_loader_discovers_all_scenes(self) -> None:
        datasets = load_project_scenes(PROJECT_ROOT)

        self.assertEqual(set(datasets), {"Box", "Entrance", "Fountain", "Statue"})
        self.assertEqual(len(datasets["Entrance"].views), 12)
        self.assertEqual(len(datasets["Statue"].views), 18)
        self.assertAlmostEqual(datasets["Statue"].views[0].intrinsics.fx, 960.0, places=6)
        self.assertAlmostEqual(datasets["Statue"].views[0].intrinsics.fov_degrees, 90.0, places=6)


if __name__ == "__main__":
    unittest.main()