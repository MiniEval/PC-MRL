import math
import time

import PySide6
import numpy as np
import threading
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore


class Visualiser:
    def __init__(self, framerate=24, width=2):
        self.framerate = framerate
        # [batch, frame, pose]
        self.mutex = threading.Lock()
        self.curr_frame = 0
        self.done = False
        self.lines = None
        self.points = None
        self.colors = None
        self.width = width

        self.show_points = False
        self.view = None

        t = threading.Thread(target=self.qt_init)
        t.start()

    def qt_init(self):
        app = pg.mkQApp("Skeleton Visualisation")
        view = gl.GLViewWidget()

        self.point_item = gl.GLLinePlotItem()
        self.line_item = gl.GLLinePlotItem()
        self.line_item.setGLOptions("opaque")
        self.grid = gl.GLGridItem()
        self.grid.setSize(5, 5, 5)
        self.grid.setSpacing(0.2, 0.2, 0.2)
        self.grid.setColor((0.0, 0.0, 0.0, 255 / 4 * 3))
        # view.addItem(self.grid)
        view.addItem(self.line_item)
        view.addItem(self.point_item)

        view.setBackgroundColor('w')

        timer = QtCore.QTimer()
        timer.timeout.connect(self._update_scene)
        timer.start(round(1000 / self.framerate))

        view.show()

        # camera_params = {'elevation': 0.0,
        #                  'azimuth': 225.0,
        #                  'distance': 9.0,
        #                  'center': PySide6.QtGui.QVector3D(0.82, -0.79, 0),
        #                  'fov': 45}
        camera_params = {'elevation': 0.0,
                         'azimuth': 180.0,
                         # 'azimuth': -90.0,
                         'distance': 12.0,
                         'center': PySide6.QtGui.QVector3D(0.0, 0.0, 1.0),
                         'fov': 45}
        view.setCameraParams(**camera_params)

        app.topLevelWindows()[0].resize(960, 540)

        self.view = view

        app.exec()

    def _update_scene(self):
        with self.mutex:
            if self.lines is not None:
                self.curr_frame = (self.curr_frame + 1) % self.lines.shape[0]
                if self.colors is None:
                    color = (0.0, 0.0, 0.0, 1.0)
                else:
                    color = self.colors[self.curr_frame]
                self.line_item.setData(pos=self.lines[self.curr_frame], color=color, width=self.width,
                                       antialias=True, mode='lines')
                self.point_item.setData(pos=self.points[self.curr_frame], color=color, width=self.width * 2,
                                        antialias=True, mode='lines')

                self.point_item.setVisible(self.show_points)

                # print(self.view.cameraParams())

    def update_data(self, heads, tails, colors=None):
        with self.mutex:
            self.lines = np.stack([heads.detach().cpu().numpy(), tails.detach().cpu().numpy()], axis=-2)
            self.lines = np.reshape(self.lines, (self.lines.shape[0], -1, 3))
            self.points = np.stack([heads.detach().cpu().numpy() - [0, 0, self.width / 140],
                                    heads.detach().cpu().numpy() + [0, 0, self.width / 140]], axis=-2)
            self.points = np.reshape(self.points, (self.points.shape[0], -1, 3))
            # self.lines -= np.mean(self.lines, axis=(0, 1), keepdims=True)

            if colors is None:
                self.colors = None
            else:
                self.colors = colors.detach().cpu().numpy()
                self.colors = np.stack([self.colors, self.colors], axis=-2)
                self.colors = np.reshape(self.colors, (self.lines.shape[0], -1, 4))

            self.curr_frame = 0

    def stop(self):
        self.done = True
