import numpy as np
from draw_util import DrawUtil
from lidar_util import imshowLocal
import math

import gym
from gym.envs.registration import register

register(
    id='vaernn-v0',
    entry_point='vaernnEnv0:vaernnEnv0'
)

lineSegments = np.array(
    [
        [[-3.5,-3.5],[ 3.5, -3.5]],
        [[-3.5,-3.5],[-3.5,  3.5]],
        [[-3.5, 3.5],[ 3.5,  3.5]],
        [[ 3.5,-3.5],[ 3.5,  3.5]],

        [[-0.9, 2.5],[-0.6, 2.5]],
        [[-0.9, 2.5],[-0.9, 0.5]],
        [[-0.6, 2.5],[-0.6, 0.5]],
        [[-0.9, 0.5],[-0.6, 0.5]],

        [[-0.9, -2.5],[-0.6, -2.5]],
        [[-0.9, -2.5],[-0.9, -0.5]],
        [[-0.6, -2.5],[-0.6, -0.5]],
        [[-0.9, -0.5],[-0.6, -0.5]],

        [[0.9, 2.5],[0.6, 2.5]],
        [[0.9, 2.5],[0.9, 0.5]],
        [[0.6, 2.5],[0.6, 0.5]],
        [[0.9, 0.5],[0.6, 0.5]],
        
        [[0.9, -2.5],[0.6, -2.5]],
        [[0.9, -2.5],[0.9, -0.5]],
        [[0.6, -2.5],[0.6, -0.5]],
        [[0.9, -0.5],[0.6, -0.5]],

        [[1.5+0.9, 2.5],[1.5+0.6, 2.5]],
        [[1.5+0.9, 2.5],[1.5+0.9, 0.5]],
        [[1.5+0.6, 2.5],[1.5+0.6, 0.5]],
        [[1.5+0.9, 0.5],[1.5+0.6, 0.5]],
        
        [[1.5+0.9, -2.5],[1.5+0.6, -2.5]],
        [[1.5+0.9, -2.5],[1.5+0.9, -0.5]],
        [[1.5+0.6, -2.5],[1.5+0.6, -0.5]],
        [[1.5+0.9, -0.5],[1.5+0.6, -0.5]],
    ]
    )

def detect_collision(points):
    large_detect  = (-0.9 - 0.25 <= points[:, 0]) & (points[:, 0] <= 2.4 + 0.25) 
    large_detect &= (-2.5 - 0.25 <= points[:, 1]) & (points[:, 1] <= 2.5 + 0.25) 

    medium_detect = (0.5 - 0.25 <= points[:, 1]) | (points[:, 1] <= -0.5 + 0.25)

    small_detect  = (points[:, 0] <= -0.6 + 0.25)
    small_detect |= (0.6 - 0.25 <= points[:, 0]) & (points[:, 0] <= 0.9 + 0.25)
    small_detect |= (2.1 - 0.25 <= points[:, 0])

    detect_index = large_detect & medium_detect & small_detect

    return points[~detect_index]

if __name__ == '__main__':
    env = gym.make('vaernn-v0')
    env.setting()

    points = 6.5 * np.random.rand(10000, 2) -3.25
    detect_points = detect_collision(points)
    yaws = (2.0 * np.random.rand(detect_points.shape[0]) -1.0) * math.pi

    maxLen = 1.
    deg_offset = 45.
    rad_offset = deg_offset*(math.pi/180.0)
    startDeg = -135. + deg_offset
    endDeg = 135. + deg_offset
    resolusion = 0.25

    draw_util = DrawUtil(maxLen)
    draw_util.update()

    for point, yaw in zip(detect_points, yaws):
        draw_util.clear()

        observe = env.observe_env(point[0], point[1], yaw)
        print(observe.shape)

        rads = np.arange(startDeg, endDeg, resolusion)*(math.pi/180.0) + rad_offset
        cos = np.cos(rads)
        sin = np.sin(rads)
        posLocal = np.array([[l*c, l*s] for l, c, s in zip(observe, cos, sin)])
        zeros = np.zeros(posLocal.shape)

        points = np.stack([posLocal, zeros], axis=1)
        draw_util.draw_lines(points, psize=0, color='blue', alpha=0.2)
        draw_util.draw_points(posLocal, psize=1)

        draw_util.draw_circles(np.array([[0, 0]]), psize=0.25/8.0)

        draw_util.update(1)
