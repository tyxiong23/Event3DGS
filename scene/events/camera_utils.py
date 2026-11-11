#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from scene.events.dataset_readers import EventCameraInfo
from scene.events.event_cameras import EventCamera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)
    if cam_info.is_train and resolution != (orig_w, orig_h):
        raise ValueError('Event supervision requires native resolution (-r 1); resize events/intrinsics and Bayer layout together in preprocessing.')

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[0] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]
    
    if cam_info.is_train == False:
        return EventCamera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device)
    else:
        prev_camera = loadCam(args, id, cam_info.prev, resolution_scale)
        return EventCamera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device,
                  is_train = True, is_colored = (not args.monochrome), event_image = cam_info.event_image, prev = prev_camera
        )


def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(idx, camera: EventCameraInfo):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    if camera.is_train == False:
        camera_entry = {
            'id' : idx,
            'img_name' : camera.image_name,
            'width' : camera.width,
            'height' : camera.height,
            'position': pos.tolist(),
            'rotation': serializable_array_2d,
            'fy' : fov2focal(camera.FovY, camera.height),
            'fx' : fov2focal(camera.FovX, camera.width),
            'is_train': camera.is_train,
        }
    else:
        prev_json = camera_to_JSON(idx, camera.prev)
        camera_entry = {
            'id' : idx,
            'img_name' : camera.image_name,
            'width' : camera.width,
            'height' : camera.height,
            'position': pos.tolist(),
            'rotation': serializable_array_2d,
            'fy' : fov2focal(camera.FovY, camera.height),
            'fx' : fov2focal(camera.FovX, camera.width),
            'is_train': camera.is_train,
            'start_t': float(camera.start_t),
            'end_t': float(camera.end_t),
            'start_idx': int(camera.start_idx),
            'end_idx': int(camera.end_idx),
            'prev': prev_json,
        }
    return camera_entry
