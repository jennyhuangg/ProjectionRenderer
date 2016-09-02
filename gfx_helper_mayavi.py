"""Setup script for 3D surface plotting with Mayavi.

By Jadrian Miles, December 2015

Just import this helper to use it:
    from gfx_helper_mayavi import *
or
    import gfx_helper_mayavi as mayavi

This will give you three useful top-level functions for setting up a figure,
drawing a surface in it, and displaying it, respectively:
  
  fig = setUpFigure()
  drawTriMesh(v, t, fig)
  showFigure(fig)

You may get some errors when you try to import this module; your environment
needs to be configured first.  I've tried to provide helpful error messages to
guide you, so just try importing and follow the instructions.
"""

from sys import prefix as __sys_prefix
if "canopy" not in __sys_prefix.lower():
  raise ImportError("Are you sure you're running the Canopy version of Python?")

import numpy as np
from numpy.linalg import norm
try:
  from mayavi import mlab
except ImportError as e:
  print "While attempting to import mayavi, I got this exception:"
  print " %%% ", e
  print "This is probably happening because Mayavi isn't installed by default"
  print "in Canopy.  To fix this issue, open up the Canopy program and click on"
  print "Package Manager.  Then search for Mayavi and install it."
  from sys import exit
  exit(-1)
except ValueError as e:
  print "While attempting to import mayavi, I got this exception:"
  print " %%% ", e
  print "This is probably happening because the GUI back-end toolkit for Python"
  print "is set to an unsupported default.  Try switching to the QT 4 toolkit."
  print "  You accomplish this by setting the ETS_TOOLKIT environment variable"
  print 'to "qt4".  To be safe, it\'s best to do this in two files in your home'
  print "directory: .bashrc and .bash_profile."
  print "  Open each of those files in your text editor (create new files if"
  print "they don't already exist) and add this line at the bottom:"
  print "    export ETS_TOOLKIT=qt4"
  print "Then exit the command line, re-open it, and try again."
  from sys import exit
  exit(-1)

def setUpFigure():
  """Sets up a Mayavi figure and returns the handle to it."""
  fig = mlab.figure(size = (800, 600), bgcolor = (1,1,1))
  fig.scene.disable_render = True
  return fig

def showFigure(fig):
  """
  Shows the given Mayavi figure.  Call this after draw calls have been made
  on the figure, since it will block until the figure is closed.
  """
  # Adjust the lighting.
  fig.scene.light_manager.lights[0].elevation = 29.0
  fig.scene.light_manager.lights[0].azimuth = 8.0
  fig.scene.light_manager.lights[0].intensity = 1.0
  fig.scene.light_manager.lights[1].elevation = -40.0
  fig.scene.light_manager.lights[1].azimuth = -60.0
  fig.scene.light_manager.lights[1].intensity = 0.6
  
  # Adjust the camera position.
  fig.scene.camera.position = [13.2, -41.2, 19.0]
  fig.scene.camera.view_up = [0, 0, 1]
  fig.scene.camera.focal_point = [0, 0, 0]
  fig.scene.camera.compute_view_plane_normal()
  
  # Turn on the X-Y-Z axis icon in the lower-left, to help orient the viewer.
  fig.scene.show_axes = True
  
  # Now finally show it!
  fig.scene.render_window.aa_frames = 8
  fig.scene.disable_render = False
  fig.scene.render()
  mlab.show()

def drawTriMesh(verts, tris=None, figure=None, **kwargs):
  """
  Draws a triangle mesh in an open Mayavi window.  Takes three positional
  arguemnts:
   - verts (3xN or 4xN): vertex array
   - tris (Mx3): triangle array
   - figure: a Mayavi figure handle
  The first two arguments may instead be passed in as a 2-tuple --- see
  example below.

  There are also several optional keyword arguments:
   - color (3-tuple): an RGB triplet to use as a color.  Default is drawn from
     a predefined color cycle.  Each color channel is a number in [0.0,1.0].
   - faces (boolean, default True): whether to draw the faces of the
     triangles.
   - cull_backfaces (boolean, default False): whether to hide triangles that
     face away from the camera.
   - edges (boolean, default False): whether to draw the edges of the
     triangles.
   - edge_color (3-tuple): an RGB triplet to use as an edge color.  Default is
     a dark grey, (0.125, 0.125, 0.125).  Each color channel is a number in
     [0.0,1.0].
   - normals (boolean, default False): whether to draw face-normal arrows. 
     The arrows will be drawn in the edge color.
   - smooth (boolean, default False): whether to shade the surface as though
     it were smoothly curved, rather than faceted.
  These arguments must be specified by name.

  Examples: If you have a 3xN vertex array V, an Mx3 triangle array T, and a
  Mayavi figure fig, here are some example calls:
   - You can use standard positional arguments:
      drawTriMesh(V, T, fig)
   - You can pass the verts and tris in a 2-tuple, and the figure will work:
      drawTriMesh( (V, T), fig )
   - You can draw the edges and specify your own face color:
      drawTriMesh(V, T, fig, color=(0.5, 1.0, 0.1), draw_edges=True)
  """
  # Get verts and tris from tuple.
  if isinstance(verts, tuple) and len(verts) == 2:
    if tris is not None:
      if figure is not None:
        raise ValueError("Too many positional arguments when passing a tuple.")
      figure = tris
    tris = verts[1]
    verts = verts[0]
  # Check type & format of verts & tris.
  if not(isinstance(verts, np.ndarray) and isinstance(tris, np.ndarray)):
    raise TypeError("verts and tris must both be Numpy arrays.")
  if isinstance(verts, np.matrix) or isinstance(tris, np.matrix):
    raise TypeError("verts and tris must be Numpy arrays, no matrices.")
  if len(verts.shape) != 2 or (verts.shape[0] != 3 and verts.shape[0] != 4):
    raise ValueError("'verts' must be 3xN or 4xN")
  if len(tris.shape) != 2 or tris.shape[1] != 3:
    raise ValueError("'tris' must be Mx3")
  if not np.issubdtype(tris.dtype, np.int):
    raise TypeError("'tris' must be an array of integers.")
  # Upcast verts to float if necessary; Mayavi won't use integer verts.
  if not np.issubdtype(verts.dtype, np.float):
    verts = verts.astype(np.float64)
  # Discard the fourth component of each vertex, if there is one.
  verts = verts[:3,:]
  
  # Get kwargs.
  color = kwargs.pop("color", None)
  if color is None:
    color = __color_cycle.next()
  faces = kwargs.pop("faces", True)
  cull_backfaces = kwargs.pop("cull_backfaces", False)
  edges = kwargs.pop("edges", False)
  edge_color = kwargs.pop("edge_color", (0.125, 0.125, 0.125))
  normals = kwargs.pop("normals", False)
  smooth = kwargs.pop("smooth", False)
  
  if not faces and not edges:
    print "Warning (drawTriMesh): you asked to not draw faces OR edges!"
  
  areas = __surfaceAreaList(verts,tris)
  if np.any(np.array(areas) < 0.0001):
    print "Warning (drawTriMesh): at least one triangle has area < 0.0001!"
  
  # Plot the surface.
  if faces:
    mesh = mlab.triangular_mesh(verts[0,:].T, verts[1,:].T, verts[2,:].T, tris,
      color=color, figure=figure)
    if smooth:
      mesh.actor.property.interpolation = 'gouraud'
    else:
      mesh.actor.property.interpolation = 'flat'
    if cull_backfaces:
      mesh.actor.property.backface_culling = True
  
  # Estimate a good edge width that won't dominate the triangles.
  edge_radius = np.mean(areas / np.array(__perimeterList(verts,tris))) * 0.2
  
  if edges:
    wires = mlab.triangular_mesh(verts[0,:].T, verts[1,:].T, verts[2,:].T, tris,
      color=edge_color, line_width=edge_radius, representation='wireframe',
      figure=figure)
  
  if normals:
    normals = np.array(__normalsList(verts, tris))
    centers = np.array(__centersList(verts, tris))
    mlab.quiver3d(centers[:,0], centers[:,1], centers[:,2],
                  normals[:,0], normals[:,1], normals[:,2],
                  color=edge_color, line_width=edge_radius*2, figure=figure)


def __normalsList(verts, tris):
  normals = [__triangleNormal(verts, tri) for tri in tris]
  return [n / norm(n) for n in normals]

def __triangleNormal(verts, tri):
  return np.cross(verts[:3,tri[1]] - verts[:3,tri[0]],
                  verts[:3,tri[2]] - verts[:3,tri[0]])

def __centersList(verts, tris):
  return [__triangleCenter(verts, tri) for tri in tris]

def __triangleCenter(verts, tri):
  return np.mean(verts[:,tri], axis=1)

def __surfaceAreaList(verts, tris):
  return [__triangleArea(verts, tri) for tri in tris]

def __triangleArea(verts, tri):
  return 0.5*norm(__triangleNormal(verts, tri))

def __perimeterList(verts, tris):
  return [__trianglePerimeter(verts, tri) for tri in tris]

def __trianglePerimeter(verts, tri):
  return (norm(verts[:,tri[1]] - verts[:,tri[0]]) +
          norm(verts[:,tri[2]] - verts[:,tri[1]]) +
          norm(verts[:,tri[0]] - verts[:,tri[2]]))

def __triMeshEdges(tris):
  # Generate all edges of all triangles, and sort the endpoint indices of each
  # edge.  Now identical edges will show up as the same pair.
  edges = np.sort(np.concatenate(
    (tris[:,(0,1)], tris[:,(1,2)], tris[:,(2,0)]), 0), 1)
  # Throw edges, as ordered pairs, into a set.  This eliminates duplicates.
  return set([(edges[i,0], edges[i,1]) for i in range(edges.shape[0])])

# Generator for a global cycle of plotting colors.
def __makeColorCycleGeneratorForMayavi():
  next_color = 0
  std_colors = [
    (0.326133, 0.586158, 0.798572),  # Blue
    (0.917628, 0.277666, 0.284002),  # Red
    (0.427885, 0.753858, 0.417907),  # Green
    (1.000000, 0.598431, 0.200000),  # Orange
    (0.680234, 0.414362, 0.719756),  # Purple
    (1.000000, 0.924668, 0.157647),  # Yellow
    (0.720000, 0.720000, 0.720000),  # Grey
    (0.968627, 0.505882, 0.749020),  # Bold pink
    (0.650980, 0.337255, 0.156863)   # Brown
  ]
  while(True):
    color = std_colors[next_color]
    next_color = (next_color + 1) % len(std_colors)
    yield color

__color_cycle = __makeColorCycleGeneratorForMayavi()
