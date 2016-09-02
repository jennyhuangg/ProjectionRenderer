import numpy as np

def drawPolyline(points, ax, linespec=None):
  """Draws points on the plane, connected by straight line segments.
  
  Arguments:
   - points (2xN numpy array): N points on the plane.
   - ax: the pyplot axes on which to plot.
   - linespec (optional string): a Matplotlib line format string. (See
     http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot)
  
  The result of this call is that all N points are drawn on the plane
  as small circles, with a straight line segment between each pair of
  consecutive points.
  """
  if len(points.shape) != 2 or points.shape[0] != 2:
    raise ValueError("'points' must be 2xN")
  if linespec is None:
    linespec = __color_cycle.next() + 'o-'
  ax.plot(points[0,:].T, points[1,:].T, linespec)


def drawPolygon(points, ax, linespec=None):
  """Draws a polygon on the plane.
  
  Arguments:
   - points (2xN numpy array): N points on the plane.
   - ax: the pyplot axes on which to plot.
   - linespec (optional string): a Matplotlib line format string. (See
     http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot)
  
  The result of this call is that all N points are drawn on the plane
  as small circles, with a straight line segment between each pair of
  consecutive points, plus an additional segment between the last and
  the first point.
  """
  if len(points.shape) != 2 or points.shape[0] != 2:
    raise ValueError("'points' must be 2xN")
  if linespec is None:
    linespec = __color_cycle.next() + 'o-'
  ax.plot(np.concatenate((points[0,:], [points[0,0]]), 1).T,
          np.concatenate((points[1,:], [points[1,0]]), 1).T, linespec)


def drawPointCloud(points, ax, color=None):
  """Draws points in space.
  
  Arguments:
   - points (3xN numpy array): N points in space.
   - ax: the pyplot axes on which to plot.
   - color (optional)
  
  Notes on arguments:
   - ax: Must be an Axes3D, which is found in mpl_toolkits.mplot3d.
     Here's how you make one:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
   - color: many values are legal:
      - None (default): the next color in the built-in color cycle.
      - "b", "g", "r", "c", "m", "y", or "k": blue, green, red, cyan,
        magenta, yellow, or black, respectively.
      - 0, 1, or 2: shade according to the x, y, or z coordinate of each
        point.
  """
  if len(points.shape) != 2 or points.shape[0] != 3:
    raise ValueError("'points' must be 3xN")
  if color == None:
    color = __color_cycle.next()
  elif color in (0, 1, 2):
    color = points[color, :]
  ax.scatter(points[0,:].T, points[1,:].T, points[2,:].T, c=color)


def drawImage(img, ax):
  """Draws a color image on a pre-existing set of axes.
  
  Arguments:
   - img (WxHx3 numpy array): Each img[x,y,:] is an RGB triplet, with
     each component in the range [0,1].
   - ax: the pyplot axes on which to plot.
  
  Note that the first index of img is treated as the X index, which
  increases from left to right.  The second index of img is treated as
  the Y index, which increases from bottom to top.
  """
  if (len(img.shape) != 3 or img.shape[2] != 3
      or np.min(img) < 0 or np.max(img) > 1):
    raise ValueError("'img' must be WxHx3, with all entries in [0,1].")
  ax.imshow(img.transpose((1,0,2)), aspect='equal', interpolation='nearest',
            origin='lower')


# Generator for a global cycle of plotting colors.
def __makeColorCycleGenerator():
  colors = 'bgrcmyk'
  next_color = 0
  while(True):
    color = colors[next_color]
    next_color = (next_color + 1) % len(colors)
    yield color

__color_cycle = __makeColorCycleGenerator()
