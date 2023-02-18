from enum import Enum
import cv2
import numpy as np

import torch
import torch.nn.functional as F
# from torch.profiler import profile, ProfilerActivity, record_function
import taichi as ti

from utils import filter2D, get_gaussian_kernel, get_time
from sgbm_core import Ti_SGBM, SGBM


class Param:
    def __init__(self):
        self.cost_mode = "sad"
        self.cost_k_size = 0
        self.blur_k_size = 0
        self.left_right = True
        self.right_left = True
        self.top_down = True
        self.down_top = True
        self.leading_diag_updown = True
        self.leading_diag_downup = True
        self.second_diag_updown = True
        self.second_diag_downup = True
        self.p1 = 10
        self.p2 = 150
        self.min_disparity = 0
        self.max_disparity = 64
        self.super_pixel_enable = False
        self.check_unique_enable = True
        self.unique_ratio = 0.1
        self.median_filter_enable = True
        self.filter_k_size= 5


class Model:
    def __init__(self, param, data_config, device):
        self.param = param
        self.data_config = data_config

        left_fn = data_config['left_fn']
        right_fn = data_config['right_fn']

        self.left = torch.from_numpy(cv2.imread(left_fn, 0)).float()
        print("load left img: ", left_fn)
        self.right = torch.from_numpy(cv2.imread(right_fn, 0)).float()
        print("load right img: ", right_fn)

        self.H, self.W = self.left.shape
        H, W = self.right.shape
        assert(self.H == H and self.W == W)

        self.device = device

    def find_center(self, img, y, x):
        pool = []
        for j in range(-1, 2):
            for i in range(-1, 2):
                if (i == 0 and j == 0):
                    continue
                pool.append(img[y+j, x+i])
        pool = sorted(pool)
        center = 0.5 * pool[3] + 0.5 * pool[4]
        return center

    def encode_img(self, img):
        # img = cv2.GaussianBlur(img.numpy(), (self.param.blur_k_size, self.param.blur_k_size), 0)
        # if self.param.cost_mode == "census" and (self.param.cost_k_size > 1):
        #     # res = torch.zeros_like(img)
        #     res = np.zeros_like(img)
        #     rad = self.param.cost_k_size // 2
        #     for y in range(rad, self.H - rad):
        #         for x in range(rad, self.W - rad):
        #             # center = img[y, x]
        #             center = self.find_center(img, y, x)
        #             val = 0
        #             for yy in range(-rad, rad+1):
        #                 for xx in range(-rad, rad+1):
        #                     if (img[y+yy, x+xx] < center):
        #                         val += 1
        #                     val = (val << 1)
        #             res[y, x] = val
        #     img = res
        # else:
        #     assert(self.param.cost_mode == "sad")

        kernel = get_gaussian_kernel(self.param.blur_k_size, channels=1).to(self.device)
        img = img.to(self.device)
        img = filter2D(img, kernel)

        if self.param.cost_mode == "census" and (self.param.cost_k_size > 1):
            res = torch.zeros_like(img).to(torch.int64)

            rad = self.param.cost_k_size // 2
            h, w = res.shape
            offsets = [(u, v) for u in range(self.param.cost_k_size) for v in range(self.param.cost_k_size) if not u==v==rad]

            img_pad = F.pad(img, pad = (rad, rad, rad, rad), mode="constant", value=0)

            for u, v in offsets:
                res = (res << 1) | (img_pad[u:u+h, v:v+w] < img)
            
            img = res
        else:
            assert(self.param.cost_mode == "sad")

        return img

    def compute_cost(self, left, right):
        # left = torch.from_numpy(left)
        # right = torch.from_numpy(right)

        disp_range = self.param.max_disparity - self.param.min_disparity
        costs = torch.zeros([self.H, self.W, disp_range], device=self.device)
        rad =  self.param.cost_k_size // 2

        for d in range(disp_range):
            right_cur = right.clone()
            right_cur[:, (rad+d):(self.W-rad)] = right[:, rad:(self.W-rad-d)]

            if self.param.cost_mode == "census":
                costs[..., d] = torch.bitwise_xor(left.to(torch.int64), right_cur.to(torch.int64))
            else:
                abs_data = torch.fabs(left - right_cur).astype(torch.uint8)
                weights = torch.ones_like([self.param.cost_k_size, self.param.cost_k_size])
                cost = filter2D(abs_data, weights)
                costs[..., d] = cost

        return costs

    def aggregate_costs(self, cost_volume):
        sgbm = Ti_SGBM(cost_volume, self.param)
        # sgbm = SGBM(cost_volume, self.param)
        aggregation_volume = sgbm.process()

        return aggregation_volume

    def index_select_2d(self, data, index):
        H, W, _ = data.shape
        res = np.zeros([H, W])
        for y in range(H):
            for x in range(W):
                res[y, x] = data[y, x, index[y, x]]
        return res

    def compute_disparity(self, volume):
        min_idx = np.argmin(volume, axis=2)

        delta = np.zeros_like(min_idx)
        _, _, C = volume.shape
        if self.param.super_pixel_enable:
            idx1 = np.clip(min_idx - 1, a_min=0, a_max=C-1)
            idx2 = np.clip(min_idx + 1, a_min=0, a_max=C-1)
            cost0 = self.index_select_2d(volume, min_idx)
            cost1 = self.index_select_2d(volume, idx1)
            cost2 = self.index_select_2d(volume, idx2)

            mask = (cost1 + cost2 - 2*cost0) != 0
            delta[mask] = (cost1[mask] - cost2[mask]) / (2 * (cost1[mask] + cost2[mask] - 2*cost0[mask]))
            
        disparity_map = min_idx + delta + self.param.min_disparity
        if self.param.check_unique_enable:
            val_min = np.min(volume, axis=2)
            H, W = min_idx.shape
            for y in range(H):
                for x in range(W):
                    volume[y, x, min_idx[y, x]] = 2**31
            val_second_min =  np.min(volume, axis=2)
            mask = (val_second_min - val_min) < (val_min * self.param.unique_ratio)
            mask = np.bitwise_and(mask, val_second_min!=0)

            disparity_map[mask] = 0

        return disparity_map

    def process(self):
        print("encoding left...")
        dawn = get_time()
        left_enc = self.encode_img(self.left)
        dust = get_time()
        print("encoded left takes: %f\n"%(dust - dawn))

        print("encoding right...")
        dawn = get_time()
        right_enc = self.encode_img(self.right)
        dust = get_time()
        print("encoded right takes: %f\n"%(dust - dawn))

        print("computing cost...")
        dawn = get_time()
        cost = self.compute_cost(left_enc, right_enc) 
        dust = get_time()
        print("computed cost takes: %f\n"%(dust - dawn))

        cost = cost.cpu()
        cost = cost.numpy().astype(np.float32)        
        print("aggregating costs...")
        dawn = get_time()
        # cost = np.load("left_cost_volume.npy")
        aggregation_volume = self.aggregate_costs(cost)
        dust = get_time()
        print("aggregation takes time: %f\n"%(dust - dawn))

        # ti.profiler.print_kernel_profiler_info('trace')
        # ti.profiler.print_scoped_profiler_info()
        # ti.profiler.memory_profiler.print_memory_profiler_info()
        # ti.reset()

        print("computing final disparity...")
        dawn = get_time()
        disparity = self.compute_disparity(aggregation_volume)
        dust = get_time()
        print("compute disparity taks time: %f\n"%(dust - dawn))

        disparity = (255.0 * disparity / self.param.max_disparity).astype(np.uint8)

        if self.param.median_filter_enable:
            disparity = cv2.medianBlur(disparity, self.param.filter_k_size)

        cv2.imwrite(self.data_config["out_fn"], disparity)
