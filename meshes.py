"""
Solution file for HW3: 3D Meshes
Comp 630 W'16 - Computer Graphics
Phillips Academy
2015-12-21

By Jadrian Miles
"""
import numpy as np

def main():
  """
  Plots several 3D shapes in different octants of space.
  """
  fig = setUpFigure()

  (cubeVerts, cubeTris) = cube()
  drawTriMesh(cubeVerts[:3,:] + np.array([[2.5, 2.5, 2.5]]).T, cubeTris,
              fig, edges=True, normals=True)

  (ellVerts, ellTris) = ell()
  drawTriMesh(ellVerts[:3,:] + np.array([[-2.5, 2.5, 2.5]]).T, ellTris,
              fig, edges=True, normals=True)

  (prismVerts, prismTris) = prism(5)
  drawTriMesh(prismVerts[:3,:] + np.array([[-2.5, -2.5, 2.5]]).T, prismTris,
              fig, edges=True, normals=True)

  (cylVerts, cylTris) = prism(15)
  drawTriMesh(cylVerts[:3,:] + np.array([[2.5, -2.5, 2.5]]).T, cylTris,
              fig, edges=True, normals=True)

  (sphVerts, sphTris) = sphere(5)
  drawTriMesh(sphVerts[:3,:] + np.array([[2.5, 2.5, -2.5]]).T, sphTris,
              fig, edges=True, normals=True)

  (torVerts, torTris) = torus(5)
  drawTriMesh(torVerts[:3,:] + np.array([[-2.5, 2.5, -2.5]]).T, torTris,
              fig, edges=True, normals=True)

  showFigure(fig)

  # After the user closes the window, we continue...

  fig = setUpFigure()
  (trfVerts, trfTris) = jmiles_shape(25)
  drawTriMesh(trfVerts[:3,:], trfTris,
              fig, edges=False, normals=False, smooth=False,
              color=(0.326133, 0.586158, 0.798572))
  showFigure(fig)


def cube():
  """
  Returns a 2-tuple (verts, tris) representing a triangle mesh of the surface
of a "unit cube".  verts is 4x8 and tris is 12x3.  See cubeVerts().
  """
  return (
    cubeVerts(),
    np.concatenate((
        fanDiskTriangles(4),
        triangleStrip(indexLoop(range(4,8)), indexLoop(range(4))),
        fanDiskTriangles(4, start=4, flip=True)
      ), 0)
  )

def cubeVerts():
  """
  Returns a 4x8 array of the eight vertices of the "unit cube".  In analogy with
the unit circle, it has "radius" 1 --- that is, the edges all have length 2.
  """
  return np.array([[ 1, 1, 1, 1],
                   [-1, 1, 1, 1],
                   [-1,-1, 1, 1],
                   [ 1,-1, 1, 1],
                   [ 1, 1,-1, 1],
                   [-1, 1,-1, 1],
                   [-1,-1,-1, 1],
                   [ 1,-1,-1, 1]]).T


def ell():
  """
  Returns a 2-tuple (verts, tris) representing a triangle mesh of the surface of
an L shape.  verts is 4x12 and tris is 20x3.  See ellVerts().
  """
  return (
    ellVerts(),
    np.concatenate((
        fanDiskTriangles(6),
        triangleStrip(indexLoop(range(6,12)), indexLoop(range(6))),
        fanDiskTriangles(6, start=6, flip=True)
      ), 0)
  )

def ellVerts():
  """
  Returns a 4x12 array of the 12 vertices of an L shape, like three cubes
attached to each other.  The edge length of each cube is 1, the whole shape is
centered on the origin, and the L lies parallel to the X-Y plane.
  """
  # Build the "front face" (in the Z=0.5 plane) of the L.
  L = np.array([[ 0, 0, 0.5, 1],
                [ 0, 1, 0.5, 1],
                [-1, 1, 0.5, 1],
                [-1,-1, 0.5, 1],
                [ 1,-1, 0.5, 1],
                [ 1, 0, 0.5, 1]]).T
  # Duplicate the front face, pushed backwards along the Z-axis, and paste
  # together.
  return np.concatenate((L, L + np.array([[0,0,-1,0]]).T), 1)


def prism(K):
  """
  Returns a 2-tuple (verts, tris) representing a triangle mesh of the surface
of a regular K-gon prism.  If K <= 6, verts is 4x(2K) and tris is (4K-4)x3;
otherwise verts is 4x(2K+2), with an extra vertex in the middle of each end-cap,
and tris is (4K-2)x3.  (Adding the extra vertex prevents extra-narrow triangles
on the end-caps.)  See prismVerts().
  """
  verts = prismVerts(K)
  if K <= 6:
    return (
      verts,
      np.concatenate((
          fanDiskTriangles(K, flip=True),
          triangleStrip(indexLoop(range(K)), indexLoop(range(K,2*K))),
          fanDiskTriangles(K, start=K)
        ), 0)
    )
  else:
    return (
      np.concatenate(([[0,1,0,1]], verts.T, [[0,-1,0,1]]), 0).T,
      np.concatenate((
          wheelDiskTriangles(K, hub=0, start=1, flip=True),
          triangleStrip(indexLoop(range(1,K+1)), indexLoop(range(K+1,2*K+1))),
          wheelDiskTriangles(K, hub=2*K+1, start=K+1)
        ), 0)
    )

def prismVerts(K):
  """
  Returns a 4x(2K) array representing vertices of a regular K-gon prism.  The
prism is centered on the origin, has height 2 along an axis parallel to the Y-
axis, and has "radius" 1: all the vertices are a distance 1 from this axis.  The
points (1,1,0) and (1,-1,0) should always be vertices of the prism.
  """
  # Build the "top cap" of the prism, which is just a sampled circle parallel to
  # the X-Z plane.
  cap = np.concatenate((
      [np.cos(np.linspace(0, 2*np.pi, K, False))],
      np.ones((1,K)),
      [np.sin(np.linspace(0, 2*np.pi, K, False))],
      np.ones((1,K))
    ), 0);
  # Like with the L, duplicate the cap, shift it, and paste the copies together.
  return np.concatenate((cap, cap + np.array([[0,-2,0,0]]).T), 1)


def sphere(K):
  """
  Returns a 2-tuple (verts, tris) representing a triangle mesh of the surface of
a unit sphere.  verts is 4x(2 + (K+1)(K+3)) and tris is ((2K+2)(K+3))x3.  See
sphereVerts().
  """
  verts = sphereVerts(K)

  s_pole = 0
  lats = K+1
  lons = K+3
  n_pole = lats*lons + 1

  s_disk = wheelDiskTriangles(lons, hub=s_pole, start=s_pole+1)
  n_disk = wheelDiskTriangles(lons, hub=n_pole, start=n_pole-lons, flip=True)
  band = 1 + triangleStrip(indexLoop(range(lons, 2*lons)),
                           indexLoop(range(0, lons)))
  # Broadcast to make "lats" offset copies of the band.
  band = band[None,:,:] + lons*np.arange(0, lats-1)[:,None,None]
  # Squish down to concatenate all those copies along the columns.
  band = np.reshape(band, (-1,3))
  # Finally, tack all the triangles together.
  tris = np.concatenate((s_disk, band, n_disk), 0)

  return (verts, tris)

def sphereVerts(K):
  """
  Returns a 4x(2 + (K+1)(K+3)) array representing vertices on the surface of the
unit sphere, centered at the origin.  The sampling on the sphere follows a
"latitude/longitude" pattern: there are K+1 lines of latitude, and K+3 lines of
longitude, equally distributed around the sphere.  There's one vertex at each
pole (2 verts total), plus one more at each lat/lon intersection (that's
(K+1)(K+3) additional verts).
  The north and south poles are at (0,1,0) and (0,-1,0), respectively, and the
"prime meridian" (which should always be included) runs between the poles
through the point (1,0,0).  (This means that your sphere should always include
at least K+3 points whose Z-coordinate is 0 and whose X-coordinate is
non-negative: the poles, plus the K+1 vertices along the prime meridian.)
  """
  # "grid_XZ" stores the X and Z coordinates of all lat/lon intersections.  The
  # inner concatenation is just the X and Z coordinates of all the intersections
  # along the equator.  We then multiply this by a cosine (in a different array
  # dimension, to leverage broadcasting) to get the points along every line of
  # latitude.  Finally, reshape squishes these down to an Nx2 array.
  grid_XZ = (
      np.concatenate((
        np.cos(np.linspace(0, 2*np.pi, K+3, False))[None, :, None],
        np.sin(np.linspace(0, 2*np.pi, K+3, False))[None, :, None]), 2) *
      np.cos(np.linspace(-np.pi/2, np.pi/2, K+2, False))[1:, None, None]
    ).reshape((-1,2))
  # The Y coordinate is constant along every line of longitude.  Again, we use
  # broadcasting across the "Layer" and "Row" dimensions to get the right number
  # of repetitions of each Y-value.  Finally, reshape squishes to Nx1.
  grid_Y = (
      np.ones((1, K+3, 1)) *
      np.sin(np.linspace(-np.pi/2, np.pi/2, K+2, False))[1:, None, None]
    ).reshape((-1,1))
  # Interleave the X, Y, and Z coordinates, plus W for homogenous coords.
  grid = np.concatenate((
      grid_XZ[:,0,None],
      grid_Y,
      grid_XZ[:,1,None],
      np.ones(grid_Y.shape)
    ), 1)
  # Add the poles and transpose.
  return np.concatenate(([[0,-1,0,1]], grid, [[0,1,0,1]]), 0).T


def torus(K):
  """
  Returns a 2-tuple (verts, tris) representing a triangle mesh of the surface of
a torus.  verts is 4x((K+3)^2) and tris is (2(K+3)^2)x3.  See torusVerts().
  """
  verts = torusVerts(K)

  N = K+3

  # One loop in the poloidal direction.
  band = triangleStrip(indexLoop(range(N, 2*N)), indexLoop(range(0, N)))
  # Broadcast to make N-1 offset copies of the loop.
  band = band[None,:,:] + N*np.arange(0, N-1)[:,None,None]
  # Squish down to concatenate all those copies along the columns.
  band = np.reshape(band, (-1,3))
  # Add one more band to close the torus up.
  tris = np.concatenate((
      band,
      triangleStrip(indexLoop(range(0, N)), indexLoop(range((N-1)*N, N*N)))
    ), 0)

  return (verts, tris)

def torusVerts(K):
  """
  Returns a 4x((K+3)^2) array representing vertices on the surface of a torus
lying parallel to the X-Y plane, centered at the origin.  The overall diameter
is 2, and the diameter of the inner hole is 2/3.  The point (1,0,0) should
always be included --- this is the intersection of two circles, other points on
which should also be included in the torus surface.  One circle is the unit
circle in the X-Y plane, and the other is perpendicular to it, in the X-Z plane,
with radius 1/3.
  """
  # "wand" is a loop around the "meat of the donut", running in the X-Z plane
  # from the outer edge of the donut, up along Z over the top, down through the
  # hole in the middle of the donut, and then back around the bottom.  (This is
  # also known as a loop in the "poloidal" direction.)
  #   The first two rows are identical, and represent the radius outward
  # perpendicular to the Z-axis (the axis of rotation as we sweep out the
  # torus).
  wand = np.concatenate((
      (np.cos(np.linspace(0, 2*np.pi, K+3, False))[None, :, None] + 2)/3,
      (np.cos(np.linspace(0, 2*np.pi, K+3, False))[None, :, None] + 2)/3,
      np.sin(np.linspace(0, 2*np.pi, K+3, False))[None, :, None]/3,
      np.ones((1, K+3, 1))
    ), 2)
  # "sweep" is a set of multipliers for the coordinates in "wand" as we copy/
  # extrude/sweep it out parallel to the X-Y plane.  Note that Z doesn't change
  # (since the axis of rotation is Z), so we just multiply it by 1 every time,
  # and of course neither does W (for homogenous coords).
  sweep = np.concatenate((
      np.cos(np.linspace(0, 2*np.pi, K+3, False))[:, None, None],
      np.sin(np.linspace(0, 2*np.pi, K+3, False))[:, None, None],
      np.ones((K+3, 1, 2))
    ), 2)
  # Alright!  Sweep that wand around, squish dimension 0 up into the rows, and
  # transpose.
  return (wand * sweep).reshape((-1,4)).T


def jmiles_shape(K):
  """
  Returns a 2-tuple (verts, tris) representing a triangle mesh of the surface of
a trefoil knot (thickened to a tube with a teardrop-shaped cross-section).
verts is  4x(3(1+3K)(K+3)) and tris is (6(1+3K)(K+3))x3.
  """
  verts = jmiles_verts(K)

  # Toroidal and poloidal sampling rates.
  M = 3*(1+3*K)
  N = K+3

  # One loop in the poloidal direction.
  band = triangleStrip(indexLoop(range(N, 2*N)), indexLoop(range(0, N)))
  # Broadcast to make M-1 offset copies of the loop.
  band = band[None,:,:] + N*np.arange(0, M-1)[:,None,None]
  # Squish down to concatenate all those copies along the columns.
  band = np.reshape(band, (-1,3))
  # Add one more band to close the torus up.
  tris = np.concatenate((
      band,
      triangleStrip(indexLoop(range(0, N)), indexLoop(range((M-1)*N, M*N)))
    ), 0)

  return (verts, tris)

def jmiles_verts(K):
  """
  Returns a 4x(3(1+3K)(K+3)) array representing vertices on the surface of a
trefoil knot thickened to a tube with a teardrop-shaped cross-section, lying
parellel to the X-Y plane, centered at the origin.  The knot fits inside the
unit sphere.
  """
  core_radius = 0.85
  tube_radius = 1 - core_radius
  pointiness = 1

  tor_sampling = 3*(1+3*K)
  pol_sampling = K+3

  # First define the core of the trefoil in spherical coordinates.  We're going
  # to treat this as analogous to the circular "core" of a torus; we define the
  # "toroidal direction" as the one that moves along this core.
  s = np.linspace(0, 2*np.pi, tor_sampling, endpoint=False)
  # Stretch and compress the spacing of samples along the "time" parameter t to
  # smooth out the positional spacing of corresponding points along the core.
  t = s - 0.5*np.sin(3*s)/3
  r = core_radius * (2+np.cos(3*t))/3
  az = 2*t
  el = np.sin(3*t)*(np.pi/4)

  # Now make a teardrop-shaped poloidal "ring" around the core.
  pol = np.linspace(0, 2*np.pi, pol_sampling, endpoint=False)
  p = tube_radius*(1-2*(np.sin(pol/2))**(1.0/pointiness))

  # Now join them in a 3-D array, with dims (toroidal, poloidal, component).
  r = r[:, None, None]
  az = az[:, None, None]
  el = el[:, None, None]

  p = p[None, :, None]
  pol = pol[None, :, None]

  x = (r + p) * np.cos(el + np.sin(pol)*tube_radius/(r+p)) * np.cos(az)
  y = (r + p) * np.cos(el + np.sin(pol)*tube_radius/(r+p)) * np.sin(az)
  z = (r + p) * np.sin(el + np.sin(pol)*tube_radius/(r+p))

  # And flatten down to (poloidal x toroidal, component).
  return np.reshape([x, y, z, np.ones(z.shape)], (4,-1))


# Helper functions!

def fanDiskTriangles(K, start=0, flip=False):
  """
  Returns a (K-2)x3 array of vertex indices for the triangulation of a K-polygon
in the plane, with indices numbered counterclockwise.  Arguments:
  - K: number of vertices in the polygon.
  - start (default 0): the starting index of the K consecutive indices around
      the polygon.
  - flip (default False): when False, triangles are oriented right-handed /
      counterclockwise; when True, they are left-handed / clockwise.
  """
  # The [:-1,:] slicing at the end eliminates the last row, which would add an
  # extra triangle between the verts on either side of the last vertex.
  return wheelDiskTriangles(K-1, hub=start, start=start+1, flip=flip)[:-1,:]

def wheelDiskTriangles(K, hub=0, start=1, flip=False):
  """
  Returns a Kx3 array of vertex indices for the triangulation of a K-polygon
in the plane, with a central "hub" vertex and K vertices in a loop around it,
numbered counterclockwise.  Arguments:
  - K: number of vertices around the outside of the polygon.
  - hub (default 0): the index of the vertex in the middle of the disk.
  - start (default 1): the starting index of the K consecutive indices around
      the polygon.
  - flip (default False): when False, triangles are oriented right-handed /
      counterclockwise; when True, they are left-handed / clockwise.
  """
  col0 = np.tile(hub, (K, 1))
  col1 = start + np.arange(K)[:,None]
  col2 = start + np.array(range(1, K) + [0])[:,None]
  if flip:
    return np.concatenate((col0, col2, col1), 1)
  else:
    return np.concatenate((col0, col1, col2), 1)

def indexLoop(idx):
  """
  Given a 1-D array or list, returns the same as a 1-D array with element #0
repeated at the end.
  """
  return np.concatenate((idx, [idx[0]]))

def triangleStrip(bot, top):
  """
  Given two 1-D arrays or lists (each of length N) of vertex indices (bot and
top), returns a (2(N-1))x3 array of indices, each row of which is a triangle in
a zigzagging strip between these parallel sets of indices:

          0  1  2  3  4  5
  top ->  *--*--*--*--*--*
          | /| /| /| /| /|
          |/ |/ |/ |/ |/ |
  bot ->  *--*--*--*--*--*
          0  1  2  3  4  5

  The triangles are oriented so that their right-hand-rule outward directions
are all facing out of the page.
  """
  bot = np.asarray(bot)
  top = np.asarray(top)
  N = len(bot)
  # There are N-1 downward-pointing triangles along the top.  The transpose
  # looks like this:
  #  [[ top[1], top[2], top[3], ... top[N-1] ]
  #   [ top[0], top[1], top[2], ... top[N-2] ]
  #   [ bot[0], bot[1], bot[2], ... bot[N-2] ]]
  jagged_top = np.array([top[1:], top[:-1], bot[:-1]]).T
  # There are N-1 upward-pointing triangles along the bottom:
  #  [[ bot[0], bot[1], bot[2], ... bot[N-2] ]
  #   [ bot[1], bot[2], bot[3], ... bot[N-1] ]
  #   [ top[1], top[2], top[3], ... top[N-1] ]]
  jagged_bot = np.array([bot[:-1], bot[1:], top[1:]]).T
  # Though it would be nice to interleave them, the order of the triangles
  # doesn't matter, so we can just stick 'em onto each other.
  return np.concatenate((jagged_top, jagged_bot), 0)

# This calls main() when the program is invoked from the command line.
if __name__ == "__main__":
  main()
