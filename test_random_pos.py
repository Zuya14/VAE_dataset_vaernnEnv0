import numpy as np
from draw_util import DrawUtil
import math

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

        [[-0.25, 0.0],[0.25, 0.0]],
        [[0.0, -0.25],[0.0, 0.25]],
    ]
    )

# def detect_collision(points):
#     large_detect  = (-0.9 - 0.25 < points[:, 0]) & (points[:, 0] < 2.4 + 0.25) 
#     large_detect &= (-2.5 - 0.25 < points[:, 1]) & (points[:, 1] < 2.5 + 0.25) 

#     medium_detect = (0.5 - 0.25 < points[:, 1]) | (points[:, 1] < -0.5 + 0.25)

#     small_detect  = (points[:, 0] < -0.6 + 0.25)
#     small_detect |= (0.6 - 0.25 < points[:, 0]) & (points[:, 0] < 0.9 + 0.25)
#     small_detect |= (2.1 - 0.25 < points[:, 0])

#     detect_index = large_detect & medium_detect & small_detect

#     return points[~detect_index]

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
    maxLen = 4.0
    draw_util = DrawUtil(maxLen)

    draw_util.drawMap(lineSegments)

    points = 6.5 * np.random.rand(10000*12, 2) -3.25
    print(points.shape)

    detect_points = detect_collision(points)
    print(detect_points.shape)

    yaws = (2.0 * np.random.rand(detect_points.shape[0]) -1.0) * math.pi
    print(yaws)

    draw_util.update()

    draw_util.draw_points(detect_points)
    # draw_util.draw_circles(detect_points, psize=0.25)
    # draw_util.draw_circles(np.zeros((1, 2)), psize=0.25, color='blue')

    draw_util.update()
    draw_util.save("test.png")