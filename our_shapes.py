"""
Custom shapes produced for HW3 by members of
Dr. Miles's Computer Graphics class,
Comp 630, W' 16 at Phillips Academy.
"""
import numpy as np

def main():
  fig = setUpFigure()

  fxns = [trefoilKnot, banana, topToy, sword, coneWedge,
          squishedBall, j_shape, doubleCone,
          doubleCone2, heartPrism]
  failed_fxns = [halfPrism, halfTorus, cone, cone2]
  ks = [11, 11, 11, 15, 11, 13, 11, 11, 11, 11]

  i = 0
  for f in fxns:
    print i
    (v, t) = f(ks[i])
    v = v + np.array([[-4.5 + 3*(i%4), -4.5 + 3*(i/4), 0]]).T
    drawTriMesh(v, t, fig, edges=False, normals=False, smooth=False)
    i += 1

  # Special calls for functions that take more than one argument.
  print i
  (v, t) = sandClock(7, 11)
  v = v + np.array([[-4.5 + 3*(i%4), -4.5 + 3*(i/4), 0]]).T
  drawTriMesh(v, t, fig, edges=False, normals=False, smooth=False)

  showFigure(fig)

# Since in HW3 we were not working in homogenous coordinates, you might find
# this function helpful to convert the vertex arrays returned by each function
# into homogeneous coordinates.
def vertsToHomogeneous(V):
  """Given a 3xN array of N vertices, returns a 4xN in homogeneous coordinates.
  """
  return np.concatenate((V, np.ones((1, V.shape[1]))), 0)

def trefoilKnot(K):
  """
  Returns a 2-tuple (verts, tris) representing a triangle mesh of the surface of
a trefoil knot (thickened to a tube with a teardrop-shaped cross-section).
verts is  3x(3(1+3K)(K+3)) and tris is (6(1+3K)(K+3))x3.
  """
  verts = trefoilKnot_verts(K)

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

def trefoilKnot_verts(K):
  """
  Returns a 3x(3(1+3K)(K+3)) array representing vertices on the surface of a
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
  return np.reshape([x, y, z], (3,-1))


def halfPrism(K):
    """ Returns a 2-tuple (verts, tris) representing a triangle mesh of the surface of
  the jpenaShape.  verts is 3x(2K) and tris is (4K-4)x3.  See halfPrism_verts().
    """
    front = fanDiskTriangles(K, start=0, flip=False)
    middle = triangleStrip(np.arange(0,K)+ K,np.arange(0,K)).T
    back = fanDiskTriangles(K,start=0,flip=True) + K
    bottom = triangleStrip([11,6],[5,0]).T
    jpenaTris = np.hstack((front,middle,back,bottom)).T
    return (halfPrism_verts(K),jpenaTris.astype(int))

def halfPrism_verts(K):
    """ returns a 3x(2K) array represnting vertices on a K-gon prism split in half.
  The prism is centered on the origin.
    """
    cap = np.concatenate((
            [np.cos(np.linspace(0, np.pi, K, False))],
            [np.sin(np.linspace(0, np.pi, K, False))],np.ones((1,K))
          ), 0);
    return np.concatenate((cap, cap + np.array([[0,0,-2]]).T), 1)


def banana(K):
  """
  Returns a 2-tuple (verts, tris) representing a triangle mesh of the surface of
  a banana-ish shape.  verts is 3x(2+K^2) and tris is (2(K^2))x3.  See
  banana_verts().
  """
  verts=banana_verts(K)
  Y=np.arange(K).reshape(-1,1)*[[K]]+np.arange(K)+[1]
  return verts, np.concatenate([
    np.fliplr(np.array([triangleStrip(Y[i], Y[i+1]) for i in range(len(Y)-1)]).reshape(-1, 3)),
    wheelDiskTriangles(K, 1+(K*K) , 1+K*(K-1), True),
    wheelDiskTriangles(K)],0)

def banana_verts(K):
  """
  Returns a 3x(2+K^2) array representing vertices on the surface of a bow/banana-
  resembling object lying generally along the X-axis, curved around the origin.
  This array is a small section of Dr. Miles's torus with two points appended
  to the ends to provide pointy banana tips.
  """
  # "wand" is a loop around the "meat of the donut", running in the X-Z plane
  # from the outer edge of the donut, up along Z over the top, down through the
  # hole in the middle of the donut, and then back around the bottom.  (This is
  # also known as a loop in the "poloidal" direction.)
  #   The first two rows are identical, and represent the radius outward
  # perpendicular to the Z-axis (the axis of rotation as we sweep out the
  # torus).
  wand = np.concatenate((
    (np.cos(np.linspace(0, 2*np.pi, K, False))[None, :, None] + 2)/3,
    (np.cos(np.linspace(0, 2*np.pi, K, False))[None, :, None] + 2)/3,
    np.sin(np.linspace(0, 2*np.pi, K, False))[None, :, None]/3
    ), 2)
  # "sweep" is a set of multipliers for the coordinates in "wand" as we copy/
  # extrude/sweep it out parallel to the X-Y plane.  Note that Z doesn't change
  # (since the axis of rotation is Z), so we just multiply it by 1 every time.
  # I only took 0 to pi on this because I only wanted half the donut.
  sweep = np.concatenate((
    np.cos(np.linspace(0, np.pi, K, False))[:, None, None],
    np.sin(np.linspace(0, np.pi, K, False))[:, None, None],
    np.ones((K, 1, 1))
    ), 2)
  # Alright!  Sweep that wand around, squish dimension 0 up into the rows, and
  # transpose.

  return np.concatenate([
    np.array([[0.5,-0.5,0]]).T,
    (wand * sweep).reshape((-1,3)).T,
    np.array([[-1,0,0]]).T],1)
    # I add vertices to both ends of the donut to create pointy banana ends.


def topToy(K):
  """
  Returns a 2-tuple of (verts, tris) representing a triangle mesh of the surface
of a spinning top. verts is 3x(4(K+3)+1), and tris is (7(K+3)+(K+1))x3. See
topToy_verts.
  """
  longs = np.arange(K+3)
  # Construct the handle
  h_top = (np.arange(K+1)[:,None] + [[0,2,1]]) * [[0,1,1]]
  handle = np.concatenate((
    (longs[:,None] + [[0,1,0]]) % (K+3) + [[0,K+3,K+3]],
    (longs[:,None] + [[0,1,1]]) % (K+3) + [[0,0,K+3]]
  ), 0)
  # Construct the body
  b_top = np.concatenate((
    (longs[:,None] + [[0,1,0]]) % (K+3) + [[0,K+3,K+3]] + K+3,
    (longs[:,None] + [[0,1,1]]) % (K+3) + [[0,0,K+3]] + K+3
  ), 0)
  body = np.concatenate((
    (longs[:,None] + [[0,1,0]]) % (K+3) + [[0,K+3,K+3]] + 2*(K+3),
    (longs[:,None] + [[0,1,1]]) % (K+3) + [[0,0,K+3]] + 2*(K+3)
  ), 0)
  bot = (longs[:,None] + [[1,0,0]]) % (K+3)*[[1,0,1]]+[[0,K+3,0]]+3*(K+3)
  return topToy_verts(K), np.concatenate((
    h_top, handle, b_top, body, bot
  ), 0)

def topToy_verts(K):
  """
  Returns a 3x(4(K+3)+1) array representing vertices of a spinning top made of
two cylinders and a cone. The smaller cylinder (the "handle") has radius 1/2 and
the wider one has radius 1 around the Y-axis. The height of the cone is 1, and
each of the two other cylinders has height 1/2. The "center" of the shape is at
the center of the boundary between the larger cylinder and the cone.
  """
  thetas = np.linspace(0, 2*np.pi, K+3, False)
  circ = np.array([np.cos(thetas), [0]*(K+3), np.sin(thetas)])
  # Create the handle
  handle_bot = circ * 0.5
  handle_top = shift(handle_bot, (0, 0.5, 0))
  handle = np.concatenate((handle_top, handle_bot), 1)
  # Create the body
  body_bot = circ
  body_top = shift(body_bot, (0, 0.5, 0))
  body = np.concatenate((body_top, body_bot), 1)
  # Return the result plus the last point
  return np.concatenate((
    shift(handle, (0, 0.5, 0)), body, [[0],[-1],[0]]
  ), 1)

def shift(verts, offs):
    return verts + np.reshape(offs, (3,1))

def sword(K):
  """
  Returns a 2-tuple (swordVertices, swordTris) representing a triangluar mesh of
  the surface of Cloud's Sword.  verts is 3x(2K+8) and tris is (2k+12)x3.
  A discription of Cloud's Sword is in sword_verts().
  """
  swordVertices = sword_verts(K)
  swordTris = np.array([[K-1, 0, 2*K-1], [2*K-1, 0, K]])
  for i in range(2, K):
      swordTris = np.concatenate((swordTris,
        np.array([[K, K + i - 1, K + i]])), 0)
  for i in range(0, K-1):
      swordTris = np.concatenate((swordTris, np.array([[i, i + 1, i + K]])), 0)
      swordTris = np.concatenate((swordTris,
        np.array([[i + 1, i + K + 1, i + K]])), 0)
  swordTris = np.concatenate((swordTris, np.array([[2*K+1, 2*K+2, 2*K],
  [2*K+2, 2*K+3, 2*K], [2*K+4, 2*K+1, 2*K], [2*K+4, 2*K+5, 2*K+1],
  [2*K+5, 2*K+2, 2*K+1], [2*K+5, 2*K+6, 2*K+2], [2*K+6, 2*K+3, 2*K+2],
  [2*K+6, 2*K+7, 2*K+3], [2*K+7, 2*K, 2*K+3], [2*K+7, 2*K+4, 2*K],
  [2*K+6, 2*K+5, 2*K+4], [2*K+7, 2*K+6, 2*K+4]])))
  return (swordVertices, swordTris)

def sword_verts(K):
  """
  Returns a 3x(2k+8) representing vertices of a rough representation of Cloud's
  Sword from the FF VII game. It has a hilt that is a cyllinder and the sword
  blade above it that is a quadralateral base and a pair of triangles that are
  not flat above.
  """
  # Build the "top cap" of the prism, which is just a sampled circle parallel to
  # the X-Z plane.
  cap = np.concatenate((
      [np.cos(np.linspace(0, 2*np.pi, K, False))/16],
      np.ones((1,K))*-.75,
      [np.sin(np.linspace(0, 2*np.pi, K, False))/16]
    ), 0);
  # Like with the L, duplicate the cap, shift it, and paste the copies together.
  hiltVerts = np.concatenate((cap, cap + np.array([[0,-.25,0]]).T), 1)
  swordVerts = np.array([[-.25, -.75, 0], [.1875, -.75, -.0625], [.25, -.75, 0],
  [.1875, -.75, .0625],
  [-.25, 1, 0], [.1875, .883333, -.0625], [.25, .9166667, 0],
  [.1875, .883333, .0625]]).T
  return np.concatenate((hiltVerts, swordVerts), 1)


def coneWedge(K):
  """
  Given some K value, this function returns a 2 tuple of vertices and triangles. The triangles (are supposed to) cover the entire object in a mesh pattern.
  """
  row1 = np.arange(0,3)
  bottom = row1 + np.arange(0,K-1)[:, None]
  bottom[:,0] = 0
  bottom = np.concatenate((bottom, [[0,K,1]]), 0)

  return (coneWedge_verts(K), np.concatenate((bottom, K+1-bottom),0))

def coneWedge_verts(K):
  """
  This function returns an array of vertices representing a quarter of a cone.
  """
  #set up all the vertices around what would be the base of the cone
  conebase = np.concatenate(([np.cos(np.linspace(0, np.pi, K, False))], np.zeros((1,K)), [np.sin(np.linspace(0, np.pi, K, False))]), 0);
  return np.concatenate((np.array([[1,0,0]]).T, conebase, np.array([[0,1,0]]).T), 1)


def squishedBall(K):
  """
 Returns a 2-tuple representing the surface of a shape much like a sphere that
has been compressed from three directions. The points on either end point in the
positive and negative Z directions. The first element  is a 3x(2+(3K+6)(2K+2))
array of verticies and the second is a (2(2K+2)(3K+6))x3 array represenging the
triange mesh.
  """
  bottomfan = wheelDiskTriangles(3*K+6,flip = True)
  topfan = wheelDiskTriangles(3*K+6,1+(2*K+2)*(3*K+6),1+(2*K+2)*(3*K+6)-(3*K+6))

  loop = indexLoop(np.arange(1,3*K+7))
  ring = triangleStrip(loop,loop + 3*K+6)

  ringadder = np.arange(0,(2*K+1)).reshape(-1,1,1)*(3*K+6)

  tube = np.concatenate((ring + ringadder), 0)

  tris = np.concatenate((bottomfan, tube, topfan), 0)

  return (squishedBall_verts(K), tris)

def squishedBall_verts(K):
  """
  Returns a 3x(2+(3K+6)(2K+2)) array of verticies lying on a shape much like a
sphere that has been compressed from three directions. The points on either end
point in the positive and negative Z directions.
  """
  zeroto2pi = np.linspace(0, 2*np.pi, 3*K+6, endpoint = False)
  zerotopi = np.linspace(0, np.pi, 2*K+3, endpoint = False)
  zerotopi = zerotopi[1:]
  sizes = np.sin(zerotopi).reshape(-1,1,1)
  sinusoid = np.linspace(0, 2*np.pi*1.5, 3*K + 6, endpoint = False)
  radii = (1.5-np.abs(np.sin(sinusoid)))/1.5

  xyplane = np.concatenate(([np.cos(zeroto2pi) * radii],
                        [np.sin(zeroto2pi) * radii]), 0) * sizes


  verts = np.concatenate((np.concatenate(xyplane,1),
                  [np.linspace(-1,1,2*K+3,endpoint=False)[1:].repeat(3*K+6)])
                  ,0)

  return np.concatenate(([[0],[0],[-1]], verts, [[0],[0],[1]]), 1)


def halfTorus(K):
  """
  Returns a 2-tuple (verts, tris) representing the triangle mesh of the surface
  of a half torus. The verts are 3x((K+3)^2 / 4) and tris are (2*(K+3)**2)x3
  """
  verts = halfTorus_verts(K)
  tris = np.zeros((2*(K+3)**2, 3), dtype=int).T
  y = 0
  for i in range (1, K+3):
    side = triangleStrip(np.arange(y+K+3, y+ 2*(K+3)), np.arange(y, y+K+3))
    tris = np.concatenate((tris, side), 1)
    y = y + K + 3

  firstCircle = fanDiskTriangles(K+3, start = 0, flip=False)
  secondCircle = fanDiskTriangles(K+3, start = (K+3)*(K+3) - K - 3, flip=True)
  tris = np.concatenate((firstCircle, secondCircle, tris), 1).T
  return (verts, tris)

def halfTorus_verts(K):
  """
  Returns a 3x((K+3)^2 / 4) array representing vertices on the surface of a half
  torus-like shape lying parallel to the X-Y plane. The ends of the torus are
  closed off by two circles.
  """
  # Set up two arrays to hold angle values.
  # theta is for placing circle around the larger circle
  theta = np.linspace(0, 1*np.pi, K+3, False)
  # phi handles the angles for each individual circle
  phi = np.linspace(0, 2*np.pi, K+3, False)

  x = (2.0/3 + 1.0/3 * np.cos(phi)) * np.reshape(np.cos(theta), (-1, 1))
  x = np.reshape(x, -1)
  y = (2.0/3 + 1.0/3 * np.cos(phi)) * np.reshape(np.sin(theta), (-1, 1))
  y = np.reshape(y, -1)
  z = 1.0/3 * np.sin(phi) * np.reshape(np.ones(K+3), (-1, 1))
  z = np.reshape(z, -1)

  torus = np.array([x, y, z])
  return torus


def j_shape(K):
  """
  Returns a 2-tuple (verts, tris) representing a triangle mesh of the surface of
a J shape.  verts is 3x(2K+14) and tris is (4K+16)x3.  See j_verts().
  """
  # Triangles for straight edge sections.
  edges = np.array([[0,1,6],
                   [6,1,5],
                   [1,2,3],
                   [1,3,4],
                   [7,0,6],
                   [7,8,0],
                   [0,8,1],
                   [8,9,1],
                   [9,2,1],
                   [2,9,10],
                   [2,10,3],
                   [10,11,4],
                   [3,10,4],
                   [6,5,13],
                   [13,5,12],
                   [13,12,7],
                   [7,12,8],
                   [9,8,10],
                   [8,11,10],
                   [7,6,13]])

  # Triangles for curve section.
  # Wheel that does not loop around.
  front = wheelDiskTriangles(K,1,14)[:-1]
  back = np.fliplr(wheelDiskTriangles(K, 8, K+14))[:-1]

  # Side triangle strip.
  sideBot = np.arange(K) + 14
  sideTop = np.arange(K) + K+14
  side = np.fliplr(triangleStrip(sideBot, sideTop))

  tris = np.concatenate((edges, front, back, side), 0)
  return (j_verts(K), tris)

def j_verts(K):
  """
  Returns a 3x(2K+14) array representing vertices on the surface of the letter
J lying parallel to the X-Y plane, centered at the origin. The shape has a
thickness of 1, height of 2, and width of 1.5. The J is made up of a quarter
cylinder at the corner with rectangular prism legs. There are K vertices that
make up the curve from the top/bottom of the quarter cylinder.
  """
  edges = np.array([[ 0.5, 1, 0.5],
                [ 0.5, -0.5, 0.5],
                [-0.5, -0.5, 0.5],
                [-0.5, -1, 0.5],
                [ 0.5, -1, 0.5],
                [ 1, -.5, 0.5],
                [1, 1, 0.5]]).T
  edges = np.concatenate((edges, edges + np.array([[0,0,-1]]).T), 1)

  # Finds equally distributed angles for quarter circle.
  angles = np.linspace(np.pi*3.0/2, 2*np.pi, K)

  # Finds respective points according to angle
  xValues = 0.5*(np.cos(angles)) + 0.5
  yValues = 0.5*np.sin(angles) - 0.5
  zValues = np.repeat([0.5],K).T
  curve = np.concatenate(([xValues], [yValues], [zValues]), 0)
  curve = np.concatenate((curve, curve + np.array([[0,0,-1]]).T), 1)

  return np.concatenate((edges, curve), 1)


def cone(K):
  """
  Returns a 2-tuple (verts, tris) representing a triangle mesh of the surface
  of a cone.  verts is 3x(K+1) and tris is (2*K-2)x3. See cone_verts().
  """
  top = fanDiskTriangles(K, start=1, flip=True)
  ring = wheelDiskTriangles(K)
  return (cone_verts(K), np.concatenate((top, ring), 1).T.astype(int))

def cone_verts(K):
  """
  Returns a 3x(K+1) array of vertex indices for the formation of a cone,
  centered at the origin and fitting with a cube of "radius" 1. First the base
  of the cone is constructed, at a height of y=1, parallel to the x-z plane.
  Then the point of the cone is constructed at (0, -1, 0). This gives the
  appearance that the cone is pointing downward (towards negative y).
  """
  base = np.concatenate((
      [np.cos(np.linspace(0, 2*np.pi, K, False))],
      np.ones((1,K)),
      [np.sin(np.linspace(0, 2*np.pi, K, False))]
    ), 0);
  point = np.array([[0, -1, 0]]).T

  return np.concatenate((point, base), 1)


def cone2(K):
  """
  Returns a 2-tuple (verts, tris) representing a triangle mesh of the surface of
a cone. Verts is 3x(K+2) and tris is (2K)x3.
  """
  top = wheelDiskTriangles(K, 0, 1, True)
  middle = wheelDiskTriangles(K, K + 1, 1, False)
  tris = np.concatenate((top, middle), 1).T

  return (cone2_verts(K), tris.astype(int))

def cone2_verts(K):
  """
  Returns a 3X(K+2) array representing the vertices of a cone with the sharp
tip pointing in the -y direction. The circular cap is located 1 unit in the
y direction above the origin parallel to the XZ plane, and is of radius 1.
The vertices will always contain the points [0,1,0] and [0,-1,0].
  """
  cap = np.concatenate((
      [np.cos(np.linspace(0, 2*np.pi, K, False))],
      np.ones((1,K)),
      [np.sin(np.linspace(0, 2*np.pi, K, False))]
    ), 0);

  return np.concatenate((np.array([[0,1,0]]).T, cap, np.array([[0,-1,0]]).T), 1)


def doubleCone(K):
  """
  Returns a 2-tuple (verts, tris) representing a triangle mesh of the surface of
a double cone. verts is 3x(K+2) and tris is (2K)x3. See doubleCone_verts().
  """
  Bot = np.concatenate((np.reshape(np.zeros(K-1, dtype=np.int32),(1,K-1)),
      np.reshape(np.array([0,1]),(2,1)) + np.arange(1,K)
      ), 0)
  Bot = np.concatenate((Bot,np.array([[0,K,1]]).T),1).T

  Top = np.concatenate(([[K+1]]+np.reshape(np.zeros(K-1, dtype=np.int32),(1,K-1)),
      np.reshape(np.array([1,0]),(2,1)) + np.arange(1,K)
      ), 0)
  Top = np.concatenate((Top,
      np.array([[K+1,1,K]]).T),1).T
  return (doubleCone_verts(K),np.concatenate((Bot, Top),0))

def doubleCone_verts(K):
  """ Returns a 3x(2+K) representing vertices on a double
  triangular prism with a K-sided base.
  """
  return np.concatenate((np.array([[0,-1,0]]).T, circleCreator(K),
      np.array([[0,1,0]]).T),1)

def circleCreator(K):
  """Returns a 3x(K) array representing points in the plane equally
distributed around the unit circle in the XZ plane.
  """
  x = np.cos(np.linspace(0,2*np.pi,K,False))
  y = np.zeros(K)
  z = np.sin(np.linspace(0,2*np.pi,K,False))
  return np.concatenate(([x],[y],[z]),0)


def doubleCone2(K):
  """Returns a 2-tuple (verts, tris) representing a triangle mesh of the surface of
two pyramids stuck together at the base.
verts is 3x(K+2) and tris is (2*K)x3.  See doubleCone2_verts().
  """
  bot_row = np.array([0,1,2], ndmin=2)
  bot = bot_row + np.arange(0,K-1)[:, None]
  bot[:,0] = 0
  bot = np.concatenate((bot, [[0,K,1]]), 0)
  return (doubleCone2_verts(K), np.concatenate((bot, K+1-bot),0))

def doubleCone2_verts(K):
  """Returns a K+2 array representing vertices of two pyramids with their K-gon
bases stuck together.  The pyramids are centered on the origin, and each has
height 1 along an axis parallel to the Y- axis, and has "radius" 1. As K
increases the shape approaches two cones stuck together at the base. K must
be at least 3
  """
  ring = np.concatenate((
        [np.cos(np.linspace(0, 2*np.pi, K, False))],
        np.zeros((1,K)),
        [np.sin(np.linspace(0, 2*np.pi, K, False))]
      ), 0);
  return np.concatenate((np.array([[0,-1,0]]).T, ring, np.array([[0,1,0]]).T), 1)


def heartPrism(K):
  """
  Returns a 2-tuple (verts, tris) representing a triangle mesh of the surface
  of a "heart prism".  verts is 3x( 2(K+3) ) and tris is (4K+10)x3.
  See heartPrism_verts().
  """
  front = fanDiskTriangles(2*(K+3), 0, True)
  sides = triangleStrip(indexLoop(np.arange(0, 2*(K+3))),
            indexLoop(np.arange(2*(K+3), 4*(K+3))))
  bottom = fanDiskTriangles(2*(K+3), 2*(K+3), False)
  tris = np.concatenate((front, sides, bottom), 0)
  return (heartPrism_verts(K), tris)

def heartPrism_verts(K):
  """
  Returns a 3x(2(K+3)) array representing vertices on the surface of a heart
  prism with its heart-shaped faces lying parallel to the X-Z plane. On the
  front heart face, the points (0, 1, -1), (1, 1, 0.5), (0, 1, 0.5), and
  (-1, 1, 0.5) are also always included. The two curved sections of each
  heart face are each represented by a semicircle with diameter 1 and
  K+3 points, including the endpoints (1, 1, 0.5) and (0, 1, 0.5) or
  (-1, 1, 0.5) and (0, 1, 0.5). The back heart face is congruent to the front
  heart face, just shifted back to y = -1.
  """
  xVals = np.concatenate((
            [0],
            0.5*np.cos(np.linspace(0, np.pi, K+2, False)) + 0.5,
            0.5*np.cos(np.linspace(0, np.pi, K+2, False)) - 0.5,
            [-1]))

  yVals = np.ones(2 + 2*(K+2))

  zVals = np.concatenate((
            [-1],
            0.5*np.sin(np.linspace(0, np.pi, K+2, False)) + 0.5,
            0.5*np.sin(np.linspace(0, np.pi, K+2, False)) + 0.5,
            [0.5]))

  front = np.concatenate(([xVals], [yVals], [zVals]), 0);
  # Like with the L, duplicate the front face, shift it, and paste the copies together.
  return np.concatenate((front, front + np.array([[0,-2,0]]).T), 1)


def sandClock(J,K):
  """
  Returns a 2-tuple (verts, tris) representing a triangle mesh of the surface
of a sand clock with J-gon on the positive y-axis and K-gon on the negative
y-axis. verts is 3x(J+K+1) and tris is (2J+2K-4)x3. See sandClock_verts().
  """
  verts = sandClock_verts(J,K)

  upBase = np.concatenate((
                [np.ones((J-2), dtype=np.int)],
                [np.arange(J-2)+3],
                [np.arange(J-2)+2]),0)

  downBase = np.concatenate((
                [(J+1)*np.ones((K-2), dtype=np.int)],
                [np.arange(K-2)+J+2],
                [np.arange(K-2)+J+3]),0)

  upSide = np.concatenate((
                [np.zeros((J), dtype=np.int)],
                [np.arange(J)+1],
                [np.concatenate((np.arange(J-1)+2,[1]),1)]),0)

  downSide = np.concatenate((
                [np.zeros((K), dtype=np.int)],
                [np.concatenate((np.arange(K-1)+J+2,[J+1]),1)],
                [np.arange(K)+J+1]),0)

  tris = np.concatenate((upBase,upSide,downBase,downSide),1).T

  return (verts,tris)

def sandClock_verts(J,K):
  """
  Returns a 3x(J+K+1) array representing vertices of a sand clock with J-gon on
the positive y-axis and K-gon on the negative y-axis. The array must consist the
origin, (0,0,0). The sandclock is centered on the origin, has height 2 along an
axis parallel to the Y-axis, and has "radius" 1: all the vertices, except the
origin (0,0,0), are a distance 1 from the Y-axis. As the prism does, J and K has
to be numbers greater than or equal to 3, and the greater they get, the bases of
the sand clock looks more like a circle.
  """

  up = np.array([np.cos(np.linspace(0, 2*np.pi, J, False)),
                (1)* np.ones((J), dtype=np.int),
                 np.sin(np.linspace(0, 2*np.pi, J, False))])

  down = np.array([np.cos(np.linspace(0, 2*np.pi, K, False)),
                  (-1)* np.ones((K), dtype=np.int),
                   np.sin(np.linspace(0, 2*np.pi, K, False))])

  return np.concatenate((np.reshape([0,0,0], (3,1)),up,down),1)


# These are Dr. Miles's reference implementations of the helper functions.
# Hopefully anyone who used them implemented them correctly.
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


if __name__ == "__main__":
  main()
