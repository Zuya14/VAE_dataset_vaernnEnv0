import numpy as np
from draw_util import DrawUtil
from lidar_util import imshowLocal
import math

import gym
from gym.envs.registration import register

import os
import datetime

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

def createSample(num):

    points = 6.5 * np.random.rand(num, 2) -3.25
    detect_points = detect_collision(points)
    yaws = (2.0 * np.random.rand(num) -1.0) * math.pi

    num = num - detect_points.shape[0]
    # print(detect_points.shape[0])

    while num > 0:
        _points = 6.5 * np.random.rand(num, 2) -3.25
        _detect_points = detect_collision(_points)
        # print(_detect_points.shape[0])
        np.append(detect_points, _detect_points)

        num = num - _detect_points.shape[0]
    
    return points, yaws

def createDataset(id, data_num):
    env = gym.make('vaernn-v0')
    env.setting()

    points, yaws = createSample(data_num)

    observes = [env.observe_env(point[0], point[1], yaw) for  point, yaw in zip(points, yaws)]

    obs = np.array(observes)
    # print(obs.shape)

    out_dir = 'data/vaernnEnv0/'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    np.save(out_dir + 'id-{}.npy'.format(id), obs)

if __name__ == '__main__':
    
    s_time = datetime.datetime.now()
    print("start:", s_time)

    for id in range(1):
        createDataset(id, data_num=10000)
        print("finish id-{}:".format(id), datetime.datetime.now())

    e_time = datetime.datetime.now()
    print("start:", s_time)
    print("end:", e_time)
    print("end-start:", e_time-s_time)
