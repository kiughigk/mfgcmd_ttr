############################################
#
# Calculate mahalanobis distance
# numpy and util4dm.py are required
# import numpy as np
# import util4dm as ut
#
#
############################################

#import sys
#sys.path.append('C:\Users\GKiuc339340273\skunk\dm')
#import util4dm as ut
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# x, y are lists
# target is list: [[x0, y0], [x1, x2], ...]

def get_prob_ellip_param (xdata, ydata, radius):

    cov = np.cov(xdata, ydata)
    vals, vecs = eigsorted(cov)     
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    w, h = 2 * radius * np.sqrt(vals)

    return [np.mean(xdata), np.mean(ydata), w, h, theta]

"""
ell = Ellipse(xy=(np.mean(xdata), np.mean(ydata)),
              width=w, height=h,
              angle=theta, color='black')
ell.set_facecolor('none')
ax.add_artist(ell)
"""


def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]






