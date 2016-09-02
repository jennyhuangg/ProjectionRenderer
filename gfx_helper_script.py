# gfx_helper_script.py --- setup script for non-interactive plotting with
# Matplotlib in scripts.
#   By Jadrian Miles, December 2015
# 
# To use, just stick the following line at the top of your script:
# 
# from gfx_helper_script import *
# 
# This will execute all the following commands in your script's global
# namespace.  Then you can do, e.g., the following:
# 
# x = np.linspace(-5, 5, 51)
# y = 0.4*x**2 - 0.8*x - 4
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1, aspect='equal')
# ax.set_xlim(-5, 5)
# ax.set_ylim(-5, 5)
# ax.grid(True)
# ax.plot(x, y)
# ax.plot(x, -y, 'g')
# plt.show()
# 

from sys import prefix
if "canopy" not in prefix.lower():
  raise ImportError("Are you sure you're running the Canopy version of Python?")

import numpy as np
import matplotlib
# Set the Matplotlib backend to Tk, to prevent hangs.
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# Turn off interactive mode, just in case.
plt.ioff()
# Load up 3D plotting tools too.
from mpl_toolkits.mplot3d import Axes3D
