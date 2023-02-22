import math
import typing as T
from typing import Optional
from functools import wraps
import math
import time

import torch
import torch.nn.functional as F


_Size2D = T.Union[int, T.Tuple[int, int]]


def _kernel_shape(shape: _Size2D):
    # H -> (H, H); (H, W) -> (H, W)
    if isinstance(shape, int):
        return (shape, shape)
    elif isinstance(shape, tuple):
        return shape

    raise Exception("Invalid Shape")


def get_gaussian_kernel(kernel_size=3, sigma=2, channels=3):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1)/2.0
    variance = sigma**2.0

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1.0/(2.0*math.pi*variance)) *\
                    torch.exp(
                        -torch.sum((xy_grid - mean)**2.0, dim=-1) /\
                        (2.0*variance)
                    )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    return gaussian_kernel


def filter2D(inp: torch.Tensor, kernel: torch.Tensor, padding: Optional[_Size2D] = None, stride: Optional[_Size2D] = None) -> torch.Tensor:
    """Apply filter on input, just like cv2.filter2D

    Args:
        inp: (H, W) or (C, H, W) or (N, C, H, W)
        kernel: (kH, kW)
        padding: (pH, pW) or None; default value is (kH//2, kW//2)
    """

    if padding is None:
        *_, kh, kw = kernel.shape
        padding = (kh//2, kw//2)

    ph, pw = _kernel_shape(padding)
    if kernel.dim() == 2:
        kh, kw = kernel.size()
        kernel = kernel.view(1, 1, kh, kw)
    
    if stride is None:
        stride = (1, 1)

    if inp.dim() == 2:
        x = inp.view(1, 1, *inp.size())
    elif inp.dim() == 3:
        C, H, W = inp.size()
        x = inp.view(C, 1, H, W)
    elif inp.dim() == 4:
        N, C, H, W = inp.size()
        x = inp.view(N*C, 1, H, W)

    x = F.conv2d(x, kernel.to(x.dtype), bias=None, stride=stride, padding=[ph, pw])

    if inp.dim() == 2:
        return x[0, 0]
    elif inp.dim() == 3:
        C, H, W = inp.size()
        return x.view(C, H, W)
    elif inp.dim() == 4:
        N, C, H, W = inp.size()
        return x.view(N, C, H, W)


def get_time():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        print(f"Function {func.__name__} starts")

        result = func(*args, **kwargs)
        
        end_time = time.perf_counter()

        total_time = end_time - start_time
        print(f'Function {func.__name__} Took {total_time:.4f} seconds')

        return result

    return timeit_wrapper