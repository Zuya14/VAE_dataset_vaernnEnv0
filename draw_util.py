import numpy as np
import matplotlib.pyplot as plt

class DrawUtil:

    def __init__(self, maxLen=None):

        self.fig, self.ax = plt.subplots()

        if maxLen:
            self.ax.set_xlim(-maxLen, maxLen)
            self.ax.set_ylim(-maxLen, maxLen)

        self.ax.set_aspect('equal')
        self.ax.grid()

        self.maxLen = maxLen
    
    def update(self, interval=0.01):
        plt.pause(interval)

    def clear(self):
        self.ax.lines.clear()
        self.ax.collections.clear()

    def draw_points(self, points, psize=1, color='red', marker='o', alpha=1):
        p = points.reshape((-1, 2))
        self.ax.scatter(p[:,0], p[:,1], s=psize, c=color, marker=marker, alpha=alpha)

    def draw_circles(self, points, psize=1, color='red', marker='o', alpha=1):
        ppi=72
        ax_length = self.ax.bbox.get_points()[1][0] - self.ax.bbox.get_points()[0][0]
        ax_point = ax_length*ppi / self.fig.dpi

        xsize = self.maxLen*2
        fact = ax_point / xsize

        # scatterのマーカーサイズは直径のポイントの二乗を描くため、実スケールの半径をポイントに変換し直径にしておく
        psize *= 2*fact
        
        p = points.reshape((-1, 2))
        self.ax.scatter(p[:,0], p[:,1], s=psize**2, c='none', edgecolors=color, marker=marker, alpha=alpha)

    def draw_lines(self, points, psize=1, color='red', linestyle='solid'):
        p = points.reshape((-1, 2))
        self.ax.plot(p[:,0], p[:,1], ms=psize, c=color, marker="o", linestyle=linestyle)

    def drawMap(self, lineSegments, color='green', linewidth=1):
        for points in lineSegments:
            s = points[0]
            e = points[1]
            self.ax.plot([s[0], e[0]],[s[1], e[1]], c=color, linewidth=linewidth)

    def save(self, name):
        self.fig.savefig(name)