from distutils.archive_util import make_archive
from enum import IntEnum
from model import Param


class Cost_mode(IntEnum):
    SAD = 0
    CENSUS = 1


class Mask_mode(IntEnum):
    MODE_DEFAULT = 0
    MODE_1x1 = 1
    MODE_3x3 = 2
    MODE_5x5 = 3
    MODE_7x7 = 4


class Process_mode(IntEnum):
    SPEED = 0
    ACCURATE = 1


# dense increasing
class Dense_mode(IntEnum):
    MODE_0 = 0
    MODE_1 = 1
    MODE_2 = 2
    MODE_3 = 3
    MODE_4 = 4
    MODE_5 = 5


class Firmware:
    def __init__(self, config):
        self.config = config

    def render(self):
        param = Param()

        param.min_disparity = self.config["min_disparity"]
        param.max_disparity = self.config["max_disparity"]

        param.left_right = True
        param.right_left = True
        param.top_down = True
        param.down_top = True
        param.filter_k_size = 5
        param.blur_k_size = 5

        param.unique_ratio = self.config["unique_ratio"]

        if self.config['process_mode'] == Process_mode.SPEED:
            param.leading_diag_updown = False
            param.leading_diag_downup = False
            param.second_diag_updown = False
            param.second_diag_downup = False
            param.median_filter_enable = False
            param.check_unique_enable = False
        else:
            param.leading_diag_updown = True
            param.leading_diag_downup = True 
            param.second_diag_updown = True 
            param.second_diag_downup = True 
            param.median_filter_enable = True 
            param.check_unique_enable = True

        if self.config['dense_mode'] == Dense_mode.MODE_0:
            param.p1 = 10
            param.p2 = 150
        elif self.config['dense_mode'] == Dense_mode.MODE_1:
            param.p1 = 15
            param.p2 = 50
        elif self.config['dense_mode'] == Dense_mode.MODE_2:
            param.p1 = 25
            param.p2 = 50
        elif self.config['dense_mode'] == Dense_mode.MODE_3:
            param.p1 = 5
            param.p2 = 80
        elif self.config['dense_mode'] == Dense_mode.MODE_4:
            param.p1 = 15
            param.p2 = 80
        elif self.config['dense_mode'] == Dense_mode.MODE_5:
            param.p1 = 25
            param.p2 = 80
        else:
            raise ValueError("invalid dense mode")

        if self.config['cost_mode'] == Cost_mode.SAD:
            param.cost_mode = "sad"
        else:
            param.cost_mode = "census"

        if self.config['mask_mode'] == Mask_mode.MODE_1x1:
            param.cost_k_size = 1
        elif self.config['mask_mode'] == Mask_mode.MODE_3x3:
            param.cost_k_size = 3
        elif self.config['mask_mode'] == Mask_mode.MODE_5x5:
            param.cost_k_size = 5
        elif self.config['mask_mode'] == Mask_mode.MODE_7x7:
            param.cost_k_size = 7
        else:
            raise ValueError("invalid mask mode")

        param.super_pixel_enable = self.config["sub_pixel_enable"]
        param.guided_filter_enable = self.config["guided_filter_enable"]

        return param