from enum import Enum
import time
import taichi as ti
import numpy as np
from multiprocessing import Pool

from utils import timeit
from share_array import make_shared_array, get_shared_array


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


@ti.data_oriented
class Ti_SGBM:
    def __init__(self, cost_volume, param):
        self.height, self.width, self.disparities = cost_volume.shape
        self.paths = Paths(param)
        print("height: ", self.height)
        print("width: ", self.width)
        print("disparities: ", self.disparities)
        print("path size: ", self.paths.size)

        self.p1 = param.p1
        self.p2 = param.p2

        self.cost_volume = cost_volume.astype(np.float32)

    @ti.kernel
    def aggregate_NS(self, minimum_cost: ti.types.ndarray(), cost_volume: ti.types.ndarray(), p1: ti.f32, p2: ti.f32):
        for x, i in ti.ndrange(self.width, (1, self.height)):
            min_cost_last = 1.0 * 2**30
            for d in range(self.disparities):
                min_cost_last = ti.min(min_cost_last, minimum_cost[i-1, x, d])

            for d in range(self.disparities):
                l1 = minimum_cost[i-1, x, d]

                d_minus = ti.max(d-1, 0) 
                d_plus = ti.min(d+1, self.disparities - 1)
                l2 = minimum_cost[i-1, x, d_minus] + p1
                l3 = minimum_cost[i-1, x, d_plus] + p1

                l4 = min_cost_last + p2

                tmp_a = ti.min(l1, l2)
                tmp_b = ti.min(l3, l4)
                tmp = ti.min(tmp_a, tmp_b)
                minimum_cost[i, x, d] = cost_volume[i, x, d] + tmp - min_cost_last

    @ti.kernel
    def aggregate_WE(self, minimum_cost: ti.types.ndarray(), cost_volume: ti.types.ndarray(), p1: ti.f32, p2: ti.f32):
        for y, i in ti.ndrange(self.height, (1, self.width)):
            min_cost_last = 1.0 * 2**30
            for d in range(self.disparities):
                min_cost_last = ti.min(min_cost_last, minimum_cost[y, i-1, d])

            for d in range(self.disparities):
                l1 = minimum_cost[y, i-1, d]

                d_minus = ti.max(d-1, 0) 
                d_plus = ti.min(d+1, self.disparities - 1)
                l2 = minimum_cost[y, i-1, d_minus] + p1
                l3 = minimum_cost[y, i-1, d_plus] + p1

                l4 = min_cost_last + p2

                tmp_a = ti.min(l1, l2)
                tmp_b = ti.min(l3, l4)
                tmp = ti.min(tmp_a, tmp_b)
                minimum_cost[y, i, d] = cost_volume[y, i, d] + tmp - min_cost_last

    @ti.kernel
    def aggregate_NW2SE(self, minimum_cost: ti.types.ndarray(), cost_volume: ti.types.ndarray(), p1: ti.f32, p2: ti.f32):
        for line in range(self.height - 1):
            for x in range(1, ti.min(self.width, self.height - 1 - line)):
                y = x + line
                min_cost_last = 1.0 * 2**30
                for d in range(self.disparities):
                    min_cost_last = ti.min(min_cost_last, minimum_cost[y-1, x-1, d])

                for d in range(self.disparities):
                    l1 = minimum_cost[y-1, x-1, d]

                    d_minus = ti.max(d-1, 0) 
                    d_plus = ti.min(d+1, self.disparities - 1)
                    l2 = minimum_cost[y-1, x-1, d_minus] + p1
                    l3 = minimum_cost[y-1, x-1, d_plus] + p1

                    l4 = min_cost_last + p2

                    tmp_a = ti.min(l1, l2)
                    tmp_b = ti.min(l3, l4)
                    tmp = ti.min(tmp_a, tmp_b)
                    minimum_cost[y, x, d] = cost_volume[y, x, d] + tmp - min_cost_last

        for line in range(self.width - 2):
            for y in range(1, ti.min(self.height, self.width - 1 - line)):
                x = y + line + 1
                min_cost_last = 1.0 * 2**30
                for d in range(self.disparities):
                    min_cost_last = ti.min(min_cost_last, minimum_cost[y-1, x-1, d])

                for d in range(self.disparities):
                    l1 = minimum_cost[y-1, x-1, d]

                    d_minus = ti.max(d-1, 0) 
                    d_plus = ti.min(d+1, self.disparities - 1)
                    l2 = minimum_cost[y-1, x-1, d_minus] + p1
                    l3 = minimum_cost[y-1, x-1, d_plus] + p1

                    l4 = min_cost_last + p2

                    tmp_a = ti.min(l1, l2)
                    tmp_b = ti.min(l3, l4)
                    tmp = ti.min(tmp_a, tmp_b)
                    minimum_cost[y, x, d] = cost_volume[y, x, d] + tmp - min_cost_last

    @timeit
    def aggregate_S(self):
        cost_volume = np.copy(self.cost_volume)
        minimum_cost = np.copy(cost_volume)
        self.aggregate_NS(minimum_cost, cost_volume, p1=self.p1, p2=self.p2)

        return minimum_cost

    @timeit
    def aggregate_N(self):
        cost_volume = np.ascontiguousarray(np.flip(np.copy(self.cost_volume), 0))
        minimum_cost = np.copy(cost_volume)
        self.aggregate_NS(minimum_cost, cost_volume, p1=self.p1, p2=self.p2)

        return np.flip(minimum_cost, 0)

    @timeit
    def aggregate_E(self):
        cost_volume = np.copy(self.cost_volume)
        minimum_cost = np.copy(cost_volume)
        self.aggregate_WE(minimum_cost, cost_volume, p1=self.p1, p2=self.p2)

        return minimum_cost

    @timeit
    def aggregate_W(self):
        cost_volume = np.ascontiguousarray(np.flip(np.copy(self.cost_volume), 1))
        minimum_cost = np.copy(cost_volume)
        self.aggregate_WE(minimum_cost, cost_volume, p1=self.p1, p2=self.p2)

        return np.flip(minimum_cost, 1)

    @timeit
    def aggregate_SE(self):
        cost_volume = np.copy(self.cost_volume)
        minimum_cost = np.copy(cost_volume)
        self.aggregate_NW2SE(minimum_cost, cost_volume, p1=self.p1, p2=self.p2)

        return minimum_cost

    @timeit
    def aggregate_NW(self):
        cost_volume = np.ascontiguousarray(np.flip(np.flip(np.copy(self.cost_volume), 0), 1))
        minimum_cost = np.copy(cost_volume)
        self.aggregate_NW2SE(minimum_cost, cost_volume, p1=self.p1, p2=self.p2)

        return np.flip(np.flip(minimum_cost, 0), 1)

    @timeit
    def aggregate_SW(self):
        cost_volume = np.ascontiguousarray(np.flip(np.copy(self.cost_volume), 1))
        minimum_cost = np.copy(cost_volume)
        self.aggregate_NW2SE(minimum_cost, cost_volume, p1=self.p1, p2=self.p2)

        return np.flip(minimum_cost, 1)

    @timeit
    def aggregate_NE(self):
        cost_volume = np.ascontiguousarray(np.flip(np.copy(self.cost_volume), 0))
        minimum_cost = np.copy(cost_volume)
        self.aggregate_NW2SE(minimum_cost, cost_volume, p1=self.p1, p2=self.p2)

        return np.flip(minimum_cost, 0)

    def process(self):
        aggregation_volume = np.zeros(shape=(self.height, self.width, self.disparities), dtype=np.float32)

        if Direction.S in self.paths.paths:
            cur = self.aggregate_S()
            aggregation_volume += cur

        if Direction.N in self.paths.paths:
            cur = self.aggregate_N()
            aggregation_volume += cur
 
        if Direction.E in self.paths.paths:
            cur = self.aggregate_E()
            aggregation_volume += cur
 
        if Direction.W in self.paths.paths:
            cur = self.aggregate_W()
            aggregation_volume += cur

        if Direction.SE in self.paths.paths:
            cur = self.aggregate_SE()
            aggregation_volume += cur
 
        if Direction.NW in self.paths.paths:
            cur = self.aggregate_NW()
            aggregation_volume += cur
 
        if Direction.SW in self.paths.paths:
            cur = self.aggregate_SW()
            aggregation_volume += cur
 
        if Direction.NE in self.paths.paths:
            cur = self.aggregate_NE()
            aggregation_volume += cur

        print("aggregation_volume mean: ", aggregation_volume.mean())
        return aggregation_volume


class SGBM:
    def __init__(self, cost_volume, param):
        self.paths = Paths(param)
        self.param = param

        self.cost_volume = cost_volume.astype(np.float32)

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

    @timeit
    def aggregate_S(self, height, width, path_id, cost_volume):
        aggregation_volume = get_shared_array("shared_aggregation")
        for x in range(0, width):
            south = cost_volume[0:height, x, :]
            aggregation_volume[:, x, :, path_id] = self.get_path_cost(south, 1, self.param)
    
    @timeit
    def aggregate_N(self, height, width, path_id, cost_volume):
        aggregation_volume = get_shared_array("shared_aggregation")
        for x in range(0, width):
            north = np.flip(cost_volume[0:height, x, :], axis=0)
            aggregation_volume[:, x, :, path_id] = np.flip(self.get_path_cost(north, 1, self.param), axis=0)

    @timeit
    def aggregate_E(self, height, width, path_id, cost_volume):
        aggregation_volume = get_shared_array("shared_aggregation")
        for y in range(0, height):
            east = cost_volume[y, 0:width, :]
            aggregation_volume[y, :, :, path_id] = self.get_path_cost(east, 1, self.param)

    @timeit
    def aggregate_W(self, height, width, path_id, cost_volume):
        aggregation_volume = get_shared_array("shared_aggregation")
        for y in range(0, height):
            west = np.flip(cost_volume[y, 0:width, :], axis=0)
            aggregation_volume[y, :, :, path_id] = np.flip(self.get_path_cost(west, 1, self.param), axis=0)

    @timeit
    def aggregate_SE(self, start, end, path_id, cost_volume):
        aggregation_volume = get_shared_array("shared_aggregation")
        for offset in range(start, end):
            south_east = cost_volume.diagonal(offset=offset).T
            dim = south_east.shape[0]
            y_se_idx, x_se_idx = self.get_indices(offset, dim, Direction.SE, None)
            aggregation_volume[y_se_idx, x_se_idx, :, path_id] = self.get_path_cost(south_east, 1, self.param)

    @timeit
    def aggregate_NW(self, start, end, path_id, cost_volume):
        aggregation_volume = get_shared_array("shared_aggregation")
        for offset in range(start, end):
            south_east = cost_volume.diagonal(offset=offset).T
            north_west = np.flip(south_east, axis=0)
            dim = south_east.shape[0]
            y_se_idx, x_se_idx = self.get_indices(offset, dim, Direction.SE, None)
            y_nw_idx = np.flip(y_se_idx, axis=0)
            x_nw_idx = np.flip(x_se_idx, axis=0)
            aggregation_volume[y_nw_idx, x_nw_idx, :, path_id] = self.get_path_cost(north_west, 1, self.param)

    @timeit
    def aggregate_SW(self, start, end, height, path_id, cost_volume):
        aggregation_volume = get_shared_array("shared_aggregation")
        for offset in range(start, end):
            south_west = np.flipud(cost_volume).diagonal(offset=offset).T
            dim = south_west.shape[0]
            y_sw_idx, x_sw_idx = self.get_indices(offset, dim, Direction.SW, height - 1)
            aggregation_volume[y_sw_idx, x_sw_idx, :, path_id] = self.get_path_cost(south_west, 1, self.param)

    @timeit
    def aggregate_NE(self, start, end, height, path_id, cost_volume):
        aggregation_volume = get_shared_array("shared_aggregation")
        for offset in range(start, end):
            south_west = np.flipud(cost_volume).diagonal(offset=offset).T
            north_east = np.flip(south_west, axis=0)
            dim = south_west.shape[0]
            y_sw_idx, x_sw_idx = self.get_indices(offset, dim, Direction.SW, height - 1)
            y_ne_idx = np.flip(y_sw_idx, axis=0)
            x_ne_idx = np.flip(x_sw_idx, axis=0)
            aggregation_volume[y_ne_idx, x_ne_idx, :, path_id] = self.get_path_cost(north_east, 1, self.param)

    def process(self):
        # numpy-style of using multi-processing to accelerate

        height, width, disparities = self.cost_volume.shape
        aggregation_volume = np.zeros(shape=(height, width, disparities, self.paths.size), dtype=np.float32)

        make_shared_array(aggregation_volume, name="shared_aggregation")
        start = -(height - 1)
        end = width - 1
        p = Pool(8)
        path_id = 0
        if Direction.S in self.paths.paths:
            p.apply_async(self.aggregate_S, args=(height, width, path_id, self.cost_volume))
            path_id += 1
        if Direction.N in self.paths.paths:
            p.apply_async(self.aggregate_N, args=(height, width, path_id, self.cost_volume))
            path_id += 1
        if Direction.E in self.paths.paths:
            p.apply_async(self.aggregate_E, args=(height, width, path_id, self.cost_volume))
            path_id += 1
        if Direction.W in self.paths.paths:
            p.apply_async(self.aggregate_W, args=(height, width, path_id, self.cost_volume))
            path_id += 1
        if Direction.SE in self.paths.paths:
            p.apply_async(self.aggregate_SE, args=(start, end, path_id, self.cost_volume))
            path_id += 1
        if Direction.NW in self.paths.paths:
            p.apply_async(self.aggregate_NW, args=(start, end, path_id, self.cost_volume))
            path_id += 1
        if Direction.SW in self.paths.paths:
            p.apply_async(self.aggregate_SW, args=(start, end, height, path_id, self.cost_volume))
            path_id += 1
        if Direction.NE in self.paths.paths:
            p.apply_async(self.aggregate_NE, args=(start, end, height, path_id, self.cost_volume))
            path_id += 1
        p.close()
        p.join()

        aggregation_volume = get_shared_array("shared_aggregation")
        aggregation_volume = np.sum(aggregation_volume, axis=3)

        return aggregation_volume


class SGBM_ref:
    def __init__(self, cost_volume, param):
        self.height, self.width, self.disparities = cost_volume.shape

        self.paths = Paths(param)

        self.p1 = param.p1
        self.p2 = param.p2

        self.cost_volume = cost_volume.astype(np.float32)

    def aggregate_line(self, minimum_cost: np.array, cost_volume: np.array, p1: float, p2: float):
        for x in range(self.width):
            for i in range(1, self.height):
                min_cost_last = np.min(minimum_cost[i-1, x, :])

                for d in range(self.disparities):
                    l1 = minimum_cost[i-1, x, d]

                    d_minus = max(d-1, 0) 
                    d_plus = min(d+1, self.disparities - 1)
                    l2 = minimum_cost[i-1, x, d_minus] + p1
                    l3 = minimum_cost[i-1, x, d_plus] + p1

                    l4 = min_cost_last + p2

                    tmp_a = min(l1, l2)
                    tmp_b = min(l3, l4)
                    tmp = min(tmp_a, tmp_b)
                    minimum_cost[i, x, d] = cost_volume[i, x, d] + tmp - min_cost_last

    def process(self, aggregation_volume):
        path_id = 0
        if Direction.S in self.paths.paths:
            print("S line naive start")
            dawn = time.time()

            cost_volume = np.copy(self.cost_volume)
            minimum_cost = np.copy(cost_volume)
            self.aggregate_line(minimum_cost, cost_volume, p1=self.p1, p2=self.p2)
            aggregation_volume[..., path_id] = minimum_cost
            path_id += 1

            dust = time.time()
            print("S line naive end, takes: ", dust - dawn)
 