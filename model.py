from enum import Enum
import cv2
import numpy as np
import time
from multiprocessing import Pool
from share_array import *

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


class Direction(Enum):
    N = 0
    NE = 1
    E = 2
    SE = 3
    S = 4
    SW = 5
    W = 6
    NW = 7


class Paths:
    def __init__(self, param):

        self.paths = []
        if param.left_right:
            self.paths.append(Direction.E)
        if param.right_left:
            self.paths.append(Direction.W)
        if param.top_down:
            self.paths.append(Direction.S)
        if param.down_top:
            self.paths.append(Direction.N)
        if param.leading_diag_updown:
            self.paths.append(Direction.SE)
        if param.leading_diag_downup:
            self.paths.append(Direction.NW)
        if param.second_diag_updown:
            self.paths.append(Direction.NE)
        if param.second_diag_downup:
            self.paths.append(Direction.SW)

        self.size = len(self.paths)

class Model:
    def __init__(self, param, data_config):
        self.param = param
        self.data_config = data_config

        left_fn = data_config['left_fn']
        right_fn = data_config['right_fn']

        self.left = cv2.imread(left_fn, 0)
        self.right = cv2.imread(right_fn, 0)

        self.H, self.W = self.left.shape
        H, W = self.right.shape
        assert(self.H == H and self.W == W)

        self.paths = Paths(param)

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
        img = cv2.GaussianBlur(img, (self.param.blur_k_size, self.param.blur_k_size), 0)

        if self.param.cost_mode == "census" and (self.param.cost_k_size > 1):
            res = np.zeros_like(img)
            rad = self.param.cost_k_size // 2
            for y in range(rad, self.H - rad):
                for x in range(rad, self.W - rad):
                    # center = img[y, x]
                    center = self.find_center(img, y, x)
                    val = 0
                    for yy in range(-rad, rad+1):
                        for xx in range(-rad, rad+1):
                            if (img[y+yy, x+xx] < center):
                                val += 1
                            val <= 1 
                    res[y, x] = val
            img = res
        else:
            assert(self.param.cost_mode == "sad")

        return img

    def compute_cost(self, left, right):
        disp_range = self.param.max_disparity - self.param.min_disparity
        costs = np.zeros([self.H, self.W, disp_range])
        rad =  self.param.cost_k_size // 2

        for d in range(disp_range):
            right_cur = right.copy()
            right_cur[:, (rad+d):(self.W-rad)] = right[:, rad:(self.W-rad-d)]

            if self.param.cost_mode == "census":
                costs[..., d] = np.bitwise_xor(np.int64(left), np.int64(right_cur))
            else:
                abs_data = np.fabs(left - right_cur).astype(np.uint8)
                weights = np.ones_like([self.param.cost_k_size, self.param.cost_k_size])
                cost = cv2.filter2D(abs_data, -1, weights)
                costs[..., d] = cost

        return costs


    def get_indices(self, offset, dim, direction, height):
        """
        for the diagonal directions (SE, SW, NW, NE), return the array of indices for the current slice_array.
        :param offset: difference with the main diagonal of the cost volume.
        :param dim: number of elements along the path.
        :param direction: current aggregation direction.
        :param height: H of the cost volume.
        :return: arrays for the y (H dimension) and x (W dimension) indices.
        """
        y_indices = []
        x_indices = []

        for i in range(0, dim):
            if direction == Direction.SE:
                if offset < 0:
                    y_indices.append(-offset + i)
                    x_indices.append(0 + i)
                else:
                    y_indices.append(0 + i)
                    x_indices.append(offset + i)

            if direction == direction.SW:
                if offset < 0:
                    y_indices.append(height + offset - i)
                    x_indices.append(0 + i)
                else:
                    y_indices.append(height - i)
                    x_indices.append(offset + i)

        return np.array(y_indices), np.array(x_indices)

    def get_path_cost(self, slice_array, offset, parameters):
        """
        part of the aggregation step, finds the minimum costs in a D x M slice_array (where M = the number of pixels in the
        given direction)
        :param slice_array: M x D array from the cost volume.
        :param offset: ignore the pixels on the border.
        :param parameters: structure containing parameters of the algorithm.
        :return: M x D array of the minimum costs for a given slice_array in a given direction.
        """
        other_dim = slice_array.shape[0]
        disparity_dim = slice_array.shape[1]

        disparities = [d for d in range(disparity_dim)] * disparity_dim
        disparities = np.array(disparities).reshape(disparity_dim, disparity_dim)

        penalties = np.zeros(shape=(disparity_dim, disparity_dim), dtype=slice_array.dtype)
        penalties[np.abs(disparities - disparities.T) == 1] = parameters.p1
        penalties[np.abs(disparities - disparities.T) > 1] = parameters.p2

        minimum_cost_path = np.zeros(shape=(other_dim, disparity_dim), dtype=slice_array.dtype)
        minimum_cost_path[offset - 1, :] = slice_array[offset - 1, :]

        for i in range(offset, other_dim):
            previous_cost = minimum_cost_path[i - 1, :]
            current_cost = slice_array[i, :]
            costs = np.repeat(previous_cost, repeats=disparity_dim, axis=0).reshape(disparity_dim, disparity_dim)
            costs = np.amin(costs + penalties, axis=0)
            minimum_cost_path[i, :] = current_cost + costs - np.amin(previous_cost)
        return minimum_cost_path

    def aggregate_S(self, height, width, path_id, cost_volume):
        print("S start")
        aggregation_volume = get_shared_array("shared_aggregation")
        dawn = time.time()
        for x in range(0, width):
            south = cost_volume[0:height, x, :]
            aggregation_volume[:, x, :, path_id] = self.get_path_cost(south, 1, self.param)
        dust = time.time()
        print("S end in %fs"%(dust - dawn))

    def aggregate_N(self, height, width, path_id, cost_volume):
        print("N start")
        aggregation_volume = get_shared_array("shared_aggregation")
        dawn = time.time()
        for x in range(0, width):
            north = np.flip(cost_volume[0:height, x, :], axis=0)
            aggregation_volume[:, x, :, path_id] = np.flip(self.get_path_cost(north, 1, self.param), axis=0)
        dust = time.time()
        print("N end in %fs"%(dust - dawn))

    def aggregate_E(self, height, width, path_id, cost_volume):
        print("E start")
        aggregation_volume = get_shared_array("shared_aggregation")
        dawn = time.time()
        for y in range(0, height):
            east = cost_volume[y, 0:width, :]
            aggregation_volume[y, :, :, path_id] = self.get_path_cost(east, 1, self.param)
        dust = time.time()
        print("E end in %fs"%(dust - dawn)) 

    def aggregate_W(self, height, width, path_id, cost_volume):
        print("W end")
        aggregation_volume = get_shared_array("shared_aggregation")
        dawn = time.time()
        for y in range(0, height):
            west = np.flip(cost_volume[y, 0:width, :], axis=0)
            aggregation_volume[y, :, :, path_id] = np.flip(self.get_path_cost(west, 1, self.param), axis=0)
        dust = time.time()
        print("W end in %fs"%(dust - dawn))

    def aggregate_SE(self, start, end, path_id, cost_volume):
        print("SE start")
        aggregation_volume = get_shared_array("shared_aggregation")
        dawn = time.time()
        for offset in range(start, end):
            south_east = cost_volume.diagonal(offset=offset).T
            dim = south_east.shape[0]
            y_se_idx, x_se_idx = self.get_indices(offset, dim, Direction.SE, None)
            aggregation_volume[y_se_idx, x_se_idx, :, path_id] = self.get_path_cost(south_east, 1, self.param)
        dust = time.time()
        print("SE end in %fs"%(dust - dawn))

    def aggregate_NW(self, start, end, path_id, cost_volume):
        print("NW start")
        aggregation_volume = get_shared_array("shared_aggregation")
        dawn = time.time()
        for offset in range(start, end):
            south_east = cost_volume.diagonal(offset=offset).T
            north_west = np.flip(south_east, axis=0)
            dim = south_east.shape[0]
            y_se_idx, x_se_idx = self.get_indices(offset, dim, Direction.SE, None)
            y_nw_idx = np.flip(y_se_idx, axis=0)
            x_nw_idx = np.flip(x_se_idx, axis=0)
            aggregation_volume[y_nw_idx, x_nw_idx, :, path_id] = self.get_path_cost(north_west, 1, self.param)
        dust = time.time()
        print("NW end in %fs"%(dust - dawn))

    def aggregate_SW(self, start, end, height, path_id, cost_volume):
        print("SW start")
        aggregation_volume = get_shared_array("shared_aggregation")
        dawn = time.time()
        for offset in range(start, end):
            south_west = np.flipud(cost_volume).diagonal(offset=offset).T
            dim = south_west.shape[0]
            y_sw_idx, x_sw_idx = self.get_indices(offset, dim, Direction.SW, height - 1)
            aggregation_volume[y_sw_idx, x_sw_idx, :, path_id] = self.get_path_cost(south_west, 1, self.param)
        dust = time.time()
        print("SW end in %fs"%(dust - dawn))

    def aggregate_NE(self, start, end, height, path_id, cost_volume):
        print("NE start")
        aggregation_volume = get_shared_array("shared_aggregation")
        dawn = time.time()
        for offset in range(start, end):
            south_west = np.flipud(cost_volume).diagonal(offset=offset).T
            north_east = np.flip(south_west, axis=0)
            dim = south_west.shape[0]
            y_sw_idx, x_sw_idx = self.get_indices(offset, dim, Direction.SW, height - 1)
            y_ne_idx = np.flip(y_sw_idx, axis=0)
            x_ne_idx = np.flip(x_sw_idx, axis=0)
            aggregation_volume[y_ne_idx, x_ne_idx, :, path_id] = self.get_path_cost(north_east, 1, self.param)
        dust = time.time()
        print("NE end in %fs"%(dust - dawn))

    def aggregate_costs(self, cost_volume):
        """
        second step of the sgm algorithm, aggregates matching costs for N possible directions (8 in this case).
        :param cost_volume: array containing the matching costs.
        :param parameters: structure containing parameters of the algorithm.
        :param paths: structure containing all directions in which to aggregate costs.
        :return: H x W x D x N array of matching cost for all defined directions.
        """
        height = cost_volume.shape[0]
        width = cost_volume.shape[1]
        disparities = cost_volume.shape[2]
        start = -(height - 1)
        end = width - 1
        path_id = 0
        aggregation_volume = np.zeros(shape=(height, width, disparities, self.paths.size), dtype=cost_volume.dtype)
        make_shared_array(aggregation_volume, name="shared_aggregation")

        p = Pool(8)
        if Direction.S in self.paths.paths:
            p.apply_async(self.aggregate_S, args=(height, width, path_id, cost_volume))
            path_id += 1
        if Direction.N in self.paths.paths:
            p.apply_async(self.aggregate_N, args=(height, width, path_id, cost_volume))
            path_id += 1
        if Direction.E in self.paths.paths:
            p.apply_async(self.aggregate_E, args=(height, width, path_id, cost_volume))
            path_id += 1
        if Direction.W in self.paths.paths:
            p.apply_async(self.aggregate_W, args=(height, width, path_id, cost_volume))
            path_id += 1
        if Direction.SE in self.paths.paths:
            p.apply_async(self.aggregate_SE, args=(start, end, path_id, cost_volume))
            path_id += 1
        if Direction.NW in self.paths.paths:
            p.apply_async(self.aggregate_NW, args=(start, end, path_id, cost_volume))
            path_id += 1
        if Direction.SW in self.paths.paths:
            p.apply_async(self.aggregate_SW, args=(start, end, path_id, height, cost_volume))
            path_id += 1
        if Direction.NE in self.paths.paths:
            p.apply_async(self.aggregate_NE, args=(start, end, path_id, height, cost_volume))
            path_id += 1
        p.close()
        p.join()

        return get_shared_array("shared_aggregation")

    def index_select_2d(self, data, index):
        H, W, _ = data.shape
        res = np.zeros([H, W])
        for y in range(H):
            for x in range(W):
                res[y, x] = data[y, x, index[y, x]]
        return res

    def compute_disparity(self, aggregation_volume):
        volume = np.sum(aggregation_volume, axis=3)
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
            mask = (val_min - val_second_min) < (val_min * self.param.unique_ratio)

            disparity_map[mask] = 0

        return disparity_map

    def process(self):
        left_enc = self.encode_img(self.left)
        right_enc = self.encode_img(self.right)

        cost = self.compute_cost(left_enc, right_enc)
        cost = cost.astype(np.uint32) 

        dawn = time.time()
        # cost = np.load("left_cost_volume.npy")
        aggregation_volume = self.aggregate_costs(cost)
        dust = time.time()
        print("aggregation takes time: %f\n"%(dust - dawn))

        disparity = self.compute_disparity(aggregation_volume)
        disparity = (255.0 * disparity / self.param.max_disparity).astype(np.uint8)

        if self.param.median_filter_enable:
            disparity = cv2.medianBlur(disparity, self.param.filter_k_size)

        # cv2.imwrite(self.data_config["out_fn"], disparity)
        cv2.imwrite("check.png", disparity)