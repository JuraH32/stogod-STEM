from __future__ import annotations

from StemGames2026_ProjectTask.pointcloud.schemas import SceneDataset
from StemGames2026_ProjectTask.pipeline.schemas import EstimatedPose
from StemGames2026_ProjectTask.pipeline.pose.base import PoseProvider


class GroundTruthPoseProvider(PoseProvider):
    """
    Returns the camera poses that are already embedded in each SceneView.
    Only applicable to datasets where pose_source == "provided" (Box, Entrance).
    """

    def can_provide(self, dataset: SceneDataset) -> bool:
        return dataset.pose_source == "provided"

    def provide(self, dataset: SceneDataset) -> list[EstimatedPose]:
        if not self.can_provide(dataset):
            raise ValueError(
                f"GroundTruthPoseProvider cannot handle dataset '{dataset.scene_name}': "
                f"pose_source is '{dataset.pose_source}', expected 'provided'."
            )
        return [
            EstimatedPose(
                view_index=view.index,
                pose=view.pose,
                confidence=1.0,
                source="ground_truth",
            )
            for view in dataset.views
        ]
