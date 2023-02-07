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
    MODE_9x9 = 5


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
    MODE_6 = 6
    MODE_7 = 7
    MODE_8 = 8
    MODE_9 = 9


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
        if self.config['process_mode'] == Process_mode.SPEED:
            param.leading_diag_updown = False
            param.leading_diag_downup = False
            param.second_diag_updown = False
            param.second_diag_downup = False
            param.median_filter_enable = False
            param.super_pixel_enable = False
            param.check_unique_enable = False
        else:
            param.leading_diag_updown = True
            param.leading_diag_downup = True 
            param.second_diag_updown = True 
            param.second_diag_downup = True 
            param.median_filter_enable = True 
            param.super_pixel_enable = True
            param.check_unique_enable = True

        if self.config['dense_mode'] == Dense_mode.MODE_0:
            param.p1 = 5
            param.p2 = 50
            param.filter_k_size = 5
            param.blur_k_size = 5
        else:
            # TODO
            pass

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
            param.cost_k_size = 9

        return param