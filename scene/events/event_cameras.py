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

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

class EventCamera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid, 
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 is_train = False,
                 event_image = None,
                 is_colored = True,
                 prev = None
                 ):
        super(EventCamera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        

        self.color_mask = None
        self.event_image = None
        self.prev = None
        self.is_train = is_train


        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width)).to(self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        # Image/event storage may be on CPU; rasterizer matrices stay on CUDA.
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        if self.is_train:
            self.event_image = torch.from_numpy(event_image).unsqueeze(0).to(self.data_device)
            self.color_mask = torch.zeros((3, self.image_height, self.image_width), dtype=torch.bool, device=self.data_device)
            # print("self.color_mask", self.color_mask.shape, self.event_image.shape)
            self.prev = prev
            if is_colored:
                self.color_mask[0, 0::2, 0::2] = 1  # r

                self.color_mask[1, 0::2, 1::2] = 1  # g
                self.color_mask[1, 1::2, 0::2] = 1  # g

                self.color_mask[2, 1::2, 1::2] = 1  # b
            else:
                self.color_mask[...] = 1
            
