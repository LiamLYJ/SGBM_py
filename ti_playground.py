import taichi as ti
import numpy as np
import time

cost_volume = np.load("left_cost_volume.npy")
print("shape: ", cost_volume.shape)
height, width, disparities = cost_volume.shape


@ti.kernel
def aggregate_NS(minimum_cost: ti.types.ndarray(), cost_volume: ti.types.ndarray(), p1: ti.f32, p2: ti.f32):
    # ti.loop_config(serialize=True)
    for x, i in ti.ndrange(width, (1, height)):
        min_cost_last = 1.0 * 2**30
        for d in range(disparities):
            min_cost_last = ti.min(min_cost_last, minimum_cost[i-1, x, d])

        for d in range(disparities):
            l1 = minimum_cost[i-1, x, d]

            d_minus = ti.max(d-1, 0) 
            d_plus = ti.min(d+1, disparities - 1)
            l2 = minimum_cost[i-1, x, d_minus] + p1
            l3 = minimum_cost[i-1, x, d_plus] + p1

            l4 = min_cost_last + p2

            tmp_a = ti.min(l1, l2)
            tmp_b = ti.min(l3, l4)
            tmp = ti.min(tmp_a, tmp_b)
            minimum_cost[i, x, d] = cost_volume[i, x, d] + tmp - min_cost_last


# ti.init(arch=ti.cuda, device_memory_fraction=0.6)
ti.init(arch=ti.cpu)

minimum_cost = np.copy(cost_volume)
minimum_cost = minimum_cost.astype(np.float32)
print(minimum_cost.mean())
start = time.time()
aggregate_NS(minimum_cost, cost_volume, p1=10.0, p2=150.0)
end = time.time()
print(minimum_cost.mean())

print("time: ", end-start)