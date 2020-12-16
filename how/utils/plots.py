"""Plotting classes"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class EpochFigure:
    """Basic figure for plotting scores across epochs

    :param str title: Figure title
    :param str ylabel: Plot's y label
    """

    def __init__(self, title, *, ylabel):
        self.fig = plt.figure()
        self.axes = self.fig.add_subplot(1, 1, 1)
        self.title = title
        self.ylabel = ylabel

    def __del__(self):
        plt.close(self.fig)

    def __getattr__(self, name):
        # Delegate method calls on self.axes
        return getattr(self.axes, name)

    def save(self, path):
        """Save figure to given path"""
        self.axes.grid(b=True, which='major', color='k', linestyle='-')
        self.axes.grid(b=True, which='minor', color='r', linestyle='-', alpha=0.2)
        self.axes.minorticks_on()
        self.axes.legend()
        self.axes.set_xlabel('epoch')
        self.axes.set_ylabel(self.ylabel)
        self.axes.set_title(self.title)
        self.fig.savefig(path)
