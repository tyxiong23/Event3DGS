# 3DGS portions: Copyright (C) 2023, Inria / GRAPHDECO. See LICENSE.md.
"""Load event supervision for training, or camera poses alone for rendering."""
import json
import logging
from pathlib import Path
import random

from scene.events.camera_utils import cameraList_from_camInfos, camera_to_JSON
from scene.events.dataset_readers import read_event_scene
from utils.system_utils import searchForMaxIteration

logger = logging.getLogger(__name__)


class EventScene:
    def __init__(self, args, gaussians, load_iteration=None, shuffle=True,
                 resolution_scales=(1.0,), load_train=True):
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        if load_iteration is not None:
            self.loaded_iter = (searchForMaxIteration(str(Path(self.model_path) / 'point_cloud'))
                                if load_iteration == -1 else load_iteration)
        info = read_event_scene(args, load_initialization=self.loaded_iter is None,
                                load_events=self.loaded_iter is None, load_train=load_train)
        if shuffle:
            random.shuffle(info.train_cameras)
            random.shuffle(info.test_cameras)
        self.cameras_extent = info.nerf_normalization['radius']
        if not self.cameras_extent > 0:
            raise ValueError('The camera trajectory must have nonzero spatial extent.')
        self.train_cameras = {}
        self.test_cameras = {}
        for scale in resolution_scales:
            self.train_cameras[scale] = cameraList_from_camInfos(info.train_cameras, scale, args)
            self.test_cameras[scale] = cameraList_from_camInfos(info.test_cameras, scale, args)
        if self.loaded_iter is not None:
            gaussians.load_ply(str(Path(self.model_path) / 'point_cloud' /
                                   f'iteration_{self.loaded_iter}' / 'point_cloud.ply'))
        else:
            Path(self.model_path).mkdir(parents=True, exist_ok=True)
            with open(Path(self.model_path) / 'cameras.json', 'w') as file:
                json.dump([camera_to_JSON(i, camera) for i, camera in
                           enumerate(info.test_cameras + info.train_cameras)], file)
            gaussians.create_from_pcd(info.point_cloud, self.cameras_extent)
        logger.info('Loaded %d train / %d test views; scene extent %.6g',
                    len(info.train_cameras), len(info.test_cameras), self.cameras_extent)

    def save(self, iteration):
        path = Path(self.model_path) / 'point_cloud' / f'iteration_{iteration}' / 'point_cloud.ply'
        self.gaussians.save_ply(str(path))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
