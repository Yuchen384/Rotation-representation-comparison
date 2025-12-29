import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.utils import make_grid

# Optional dependency: pytorch3d_pose_refiner.
# For the purposes of simple DRR rendering in this project we only need
# InterIntensifierCalibration when using non-'bs' planes. To make the renderer
# usable without that external package, we fall back to lightweight stubs.
try:
    from pytorch3d_pose_refiner.tools.losses import GeodesicLossOrtho6d, WeightedL1Loss
    from pytorch3d_pose_refiner.tools.rotation_conversions import pose_to_tmat
    from pytorch3d_pose_refiner.Pytorch3d_dupla_pose_refiner import InterIntensifierCalibration
except ImportError:  # pragma: no cover - optional dependency
    GeodesicLossOrtho6d = None
    WeightedL1Loss = None
    pose_to_tmat = None

    class InterIntensifierCalibration:  # type: ignore
        def __init__(self):
            # Minimal stub; real values only needed for non-'bs' plane usage.
            self.fs_pos = None
            self.fs_normal_vec = None
            self.fs_vertical_vec = None

        def load_default_values(self):
            raise RuntimeError(
                "InterIntensifierCalibration is not available because "
                "pytorch3d_pose_refiner is not installed."
            )

from dupla_renderers.pytorch3d import AnatomyCT, AnatomySTL, Camera, CTRenderer, STLRenderer, Scene
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class simple_renderer():
    def __init__(self,
                 femur_path=None,
                 tibia_path=None,
                 renderer_type='STL',
                 output_type='6d',
                 normalize_translation=False,
                 binary=False,
                 load_anatomy=True,
                 cal_pixel_size=0.28,
                 cal_principal_point_h=0.0,
                 cal_principal_point_v=0.0,
                 cal_focal_length=980.0,
                 screen_size=1000,
                 is_yours=True,
                 rotation_z=False,
                 plane='bs'):
        super(simple_renderer, self).__init__()
        self.output_type = output_type
        # load model for rendering:
        self.femur_path = femur_path
        self.tibia_path = tibia_path
        self.rotation_z = rotation_z
        # calibration parameters
        self.cal_pixel_size = cal_pixel_size
        self.cal_principal_point_h = cal_principal_point_h
        self.cal_principal_point_v = cal_principal_point_v
        self.cal_focal_length = cal_focal_length
        # camera parameters
        self.screen_size = screen_size
        self.plane = plane
        self.is_yours = is_yours  # switch between your coordinate system and their coordinate system
        if load_anatomy:
            if self.femur_path is None and self.tibia_path is None:
                raise ValueError("At least one of femur_path or tibia_path must be provided when load_anatomy=True.")

            if renderer_type == 'STL':
                self.femur_ct = AnatomySTL.load_data(self.femur_path) if self.femur_path is not None else None
                self.tibia_ct = AnatomySTL.load_data(self.tibia_path) if self.tibia_path is not None else None
            else:
                self.femur_ct = AnatomyCT.load_data(self.femur_path) if self.femur_path is not None else None
                self.tibia_ct = AnatomyCT.load_data(self.tibia_path) if self.tibia_path is not None else None
        else:
            self.femur_ct = None
            self.tibia_ct = None

        self.normalize_translation = normalize_translation
        self.binary = binary
        if self.binary:
            self.binary = 1e-5

        # Renderer and scene components should be initialized here
        if load_anatomy:
            self.renderer_type = renderer_type
            renderer, cam, their_world_to_yours = self._initialize_renderer_and_scene()
            self.renderer = renderer
            self.cam = cam
            self.their_world_to_yours = their_world_to_yours.to(device)

    def _initialize_renderer_and_scene(self):
        """Initialize the renderer based on the provided calibrations and camera angles."""
        Intensifiers = InterIntensifierCalibration()
        if self.plane == 'bs':
            cam = Camera(
                "camera_1",
                (0, 0, 0),
                (0, 0, 1),
                (0, 1, 0),
                self.screen_size * self.cal_pixel_size,
                self.screen_size * self.cal_pixel_size,
                self.cal_principal_point_h,
                self.cal_principal_point_v,
                self.cal_focal_length,
            )
        else:
            Intensifiers.load_default_values()
            cam = Camera(
                "camera_2",
                screen_center_poses=Intensifiers.fs_pos,
                screen_normals=Intensifiers.fs_normal_vec,
                screen_verticals=Intensifiers.fs_vertical_vec,
                screen_sizes_h=self.screen_size * self.cal_pixel_size,
                screen_sizes_v=self.screen_size * self.cal_pixel_size,
                principal_points_h=self.cal_principal_point_h,
                principal_points_v=self.cal_principal_point_v,
                focal_lengths=self.cal_focal_length,
            )

        their_world_to_yours = torch.tensor(
            [
                [1, 0, 0, self.cal_principal_point_h],
                [0, 1, 0, self.cal_principal_point_v],
                [0, 0, 1, self.cal_focal_length],
                [0, 0, 0, 1],
            ],
            dtype=torch.float32,
        )[None]
        if self.rotation_z:
            self.rotation_about_z = torch.tensor(
                [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], device=device,
                dtype=torch.float32,
            )[None]
        else:
            self.rotation_about_z = torch.eye(4, device=device, dtype=torch.float32)[None]

        scene_sipla = Scene()
        if self.femur_ct is not None:
            scene_sipla.add_anatomies(self.femur_ct)
        if self.tibia_ct is not None:
            scene_sipla.add_anatomies(self.tibia_ct)
        scene_sipla.add_cameras(cam)

        renderer = CTRenderer(device="cuda") if self.renderer_type == 'CT' else STLRenderer(device="cuda")
        renderer.bind_scene(scene_sipla)

        return renderer, cam, their_world_to_yours

    def generate_virtual_xray_STL(self, tmat_femur, tmat_tibia, img_resolution_width, img_resolution_height):
        tmat_femur, tmat_tibia = tmat_femur.to(device), tmat_tibia.to(device)
        if self.femur_ct is not None:
            self.set_model_matrix(self.femur_ct, tmat_femur, self.their_world_to_yours)
        if self.tibia_ct is not None:
            self.set_model_matrix(self.tibia_ct, tmat_tibia, self.their_world_to_yours)
        try:
            foreground_efficient_renderer = self.renderer.render(
                0, img_resolution_width, img_resolution_height
            )[:, :, :].detach().cpu().numpy()
        except torch._C._LinAlgError as e:
            print(f"Matrix inversion failed: {e}")
            print(f"Camera transformation matrix: {self.cam.tmats}")
            return np.zeros((img_resolution_width, img_resolution_height))  # return a blank image on failure

        assert np.max(foreground_efficient_renderer) <= 1.0, rf'{np.max(foreground_efficient_renderer)}'
        assert np.min(foreground_efficient_renderer) >= 0.0, rf'{np.min(foreground_efficient_renderer)}'
        return 1 - foreground_efficient_renderer

    def generate_virtual_xray_CT(self, tmat_femur, tmat_tibia, img_resolution_width, img_resolution_height, binary=False):
        tmat_femur, tmat_tibia = tmat_femur.to(device), tmat_tibia.to(device)
        # breakpoint()
        if self.femur_ct is not None:
            self.set_model_matrix(self.femur_ct, tmat_femur, self.their_world_to_yours)
        if self.tibia_ct is not None:
            self.set_model_matrix(self.tibia_ct, tmat_tibia, self.their_world_to_yours)
        try:
            foreground_efficient_renderer = self.renderer.render_efficient_memory(
                0, img_resolution_width, img_resolution_height, binary=binary
            )[:, :, :].detach().cpu().numpy()
        except torch._C._LinAlgError as e:
            print(f"Matrix inversion failed: {e}")
            print(f"Camera transformation matrix: {self.cam.tmats}")
            return np.zeros((img_resolution_width, img_resolution_height))  # return an empty image

        assert np.max(foreground_efficient_renderer) <= 1.0, rf'{np.max(foreground_efficient_renderer)}'
        assert np.min(foreground_efficient_renderer) >= 0.0, rf'{np.min(foreground_efficient_renderer)}'
        return 1 - foreground_efficient_renderer

    def set_model_matrix(self, ct_data, tmat, their_world_to_yours):
        ct_data.set_model_matrix(torch.matmul(tmat, self.rotation_about_z),
                                 is_yours=self.is_yours, theirs_to_yours=their_world_to_yours)
