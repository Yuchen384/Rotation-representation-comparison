"""
Utilities for generating synthetic DRR datasets from CT volumes and true poses.

Submodules:
- volumes: NIfTI loading, cropping, and volume-to-DRR projection.
- poses: loading true poses from CSV and sampling noisy poses.
- scene: building a joint 3D scene in world/camera coordinates from femur+tibia.
"""


