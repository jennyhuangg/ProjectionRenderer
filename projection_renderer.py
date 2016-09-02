"""
HW5: Projection Renderer
Comp 630 W'16 - Computer Graphics
Phillips Academy
2015-1-25

By Jenny Huang
"""
import numpy as np
import camera

def perspectiveView(verts, inst_xform, world_to_view):
  """Transforms object-space points to canonical view coordinates.

  Arguments:
   - verts (4xN): object-space homogeneous coords for N vertices.
   - inst_xform (4x4): object-to-world instance xform (from scenegraph).
   - world_to_view (4x4): world-to-canonical view transform (from camera).

  Returns a 4xN array of homogeneous coordinates (with w=1) in the canonical
  view volume: -1 <= x,y,z <= 1 for all vertices whose world-space coords
  were in the view frustum of the camera.
  """
  cam_verts = world_to_view.dot(inst_xform).dot(verts)
  cam_verts = cam_verts/1.0/cam_verts[3]
  return cam_verts

def plotLines(scene, camera, width, height):
  """Plots a wireframe image of the camera's view of a scene.

  This function creates a new Matplotlib figure and plots lines in it,
  showing a wireframe representation of the camera's view of the given
  scene.  It has no return value.

  Arguments:
   - scene (RootNode object): the root node of a scenegraph.
   - camera (Camera object): the camera whose view is to be rendered.
   - width: the width of the rendered scene.
   - height: the height of the rendered scene.

  Note that if the ratio (width/height) doesn't match
  camera.naturalAspectRatio(), then the resulting image will be
  distorted.
  """
  import gfx_helper_script as helper

  instances = scene.getCompositeTransforms()
  world_to_view = camera.worldToCanonicalViewXform()

  fig = helper.plt.figure()
  ax = fig.add_subplot(1, 1, 1, aspect='equal')
  ax.set_xlim(-width/2.0, width/2.0)
  ax.set_ylim(-height/2.0, height/2.0)

  for (inst_xform, node, surf) in instances:
    (verts, tris) = node.mesh
    cam_verts = perspectiveView(verts, inst_xform, world_to_view)

    # Now let's make a 4xTx2 array cam_tris.  cam_tris[i,t,:] are the image-
    # space (x,y) coordinates of vertex i on triangle t, with i=3 representing
    # the same point as i=0.  This lets us do math on the triangles directly.
    #   We accomplish this by using tris to index into cam_verts, repeating
    # vertex #0 in each triangle.  The transpose rearranges the dimensions in
    # the order that we want.
    cam_tris = cam_verts[:2, tris[:, [0,1,2,0]]].transpose((2,1,0))

    # Remove triangles oriented clockwise in the image plane: these are the
    # back-faces, and we don't want to draw them.
    cam_tris = cam_tris[:, triangleNormals(cam_tris) > 0, :]

    # Scale cam_tris to the image size that the user requested.
    cam_tris = cam_tris * np.array([width, height]).reshape((1,1,2))/2.0

    # When ax.plot is given two MxN arrays as its arguments, it sees these as
    # the X and Y coordinates of N different curves, each of which is piecewise-
    # linear connecting M points.  We exploit this to draw all T triangles
    # in one call.
    ax.plot(cam_tris[:,:,0], cam_tris[:,:,1], color=surf.color)

  # Once we're done plotting all the triangles for all the instances...
  helper.plt.show()

def renderRaster(scene, camera, width, height):
  """Renders a raster image of the camera's view of a scene.

  Arguments:
   - scene (RootNode object): the root node of a scenegraph.
   - camera (Camera object): the camera whose view is to be rendered.
   - width: the width (number of pixels) of the rendered scene.
   - height: the height (number of pixels) of the rendered scene.

  Note that if the ratio (width/height) doesn't match
  camera.naturalAspectRatio(), then the resulting image will be
  distorted.

  Returns a (width)x(height)x3 numpy array: a color image suitable
  for plotting with gfx_helper_plotting.drawImage().
  """
  instances = scene.getCompositeTransforms()
  world_to_camera = camera.worldToCameraCentricXform()

  # Get vertices, colors, and normals of all triangles, transformed into the
  # camera-centric coordinate system.  For T triangles overall, the shapes are:
  #  - tri_verts: 3xTx4 (3 verts per triangle, T triangles, 4 coords per vert).
  #  - colors: Tx3 (RGB triplet for each triangle).
  #  - normals: Tx3 (<x,y,z> vector for each triangle).  Note that the length
  #    of each normal vector is equal to twice the area of that triangle.
  (tri_verts, colors, normals) = allTriData(instances, world_to_camera)

  # Since we're in a camera-centric coordinate system, only a triangle whose
  # normal vector has a positive Z component is facing toward the camera.
  # frontfaces will therefore be a length-T 1-D array of booleans, where an
  # entry is True only if that triangle faces toward the camera.
  frontfaces = normals[:,2] > 0

  # Use boolean indexing to remove back-facing triangles from tri_verts,
  # colors, and normals.
  tri_verts = tri_verts[:, frontfaces, :]
  colors = colors[frontfaces, :]
  normals = normals[frontfaces, :]

  # Now we have fewer triangles to render.
  num_tris = tri_verts.shape[1]
  # "There are", num_tris, "front-facing triangles in the scene."

  # Apply view-angle-based shading to the triangle colors.  Each color
  # should get multiplied by a scaling factor equal to the cosine of the angle
  # between the incoming view vector and the normal vector of the triangle.
  # (Note that you don't need to call a cosine function to compute this... think
  # about vector math, work it out symbolically, and simplify your solution.
  # The correct answer can be stated very concisely.)
  z = np.array([[0, 0, 1]])
  for i in range(num_tris):
    shading = z.dot(normals[i])/np.linalg.norm(normals[i])
    colors[i] = colors[i]*shading

  # Transform all the triangle vertices into the canonical view space.
  # Remember what that means about the resulting coordinates of vertices that
  # fall within the camera's view frustum.
  #   After this operation, tri_verts should be 3x(num_tris)x4, with W=1 for all
  # vertices.
  tri = tri_verts.reshape(-1, 4).T
  tri = camera.perspectiveNormalizationXform().dot(tri)
  tri = tri/1.0/tri[3]
  tri_verts = tri.T.reshape(3, -1, 4)

  # Create the empty image (initialized to all white) and the z-buffer.
  #   Remember that the first component in these images corresponds to the X
  # position, and the second component corresponds to Y.  So img should be
  # (width)x(height)x3, and z_buf should be (width)x(height).
  #   Given how z-buffering works, what should the initial value in the z-buffer
  # be for each pixel?
  img = np.ones((width, height, 3))
  z_buf = np.ones((width, height)) * -1

  # Rasterize each triangle.
  for t in range(num_tris):
    rasterizeTriangle(tri_verts[:, t, :], colors[t, :], img, z_buf)

  return img

def rasterizeTriangle(verts, color, img, z_buf):
  """Paints a triangle onto an image with z-buffering.

  Arguments:
   - verts (3x4): the 4-D coordinates (in canonical view space) of the three
     vertices in this triangle.
   - color (3-element 1-D array): the perceived color of this triangle.
   - img (WxHx3): the image to draw into.
   - z_buf (WxH): the z-buffer.

  The image dimensions imply a rasterization of the triangle into some number
  of fragments, each of which cooresponds to an image position (a,b).  This
  function has no return value, but instead (potentially) modifies img and
  z_buf.  Each fragment has a canonical z-coordinate Z.  For each framgent:
   - If Z is greater than the value formerly at z_buf[a,b], then
       img[a,b,:] = color and z_buf[a,b] = Z.
   - Otherwise, img[a,b,:] and z_buf[a,b] are left unchanged.
  """
  (W,H) = z_buf.shape

  # Define the mapping between canonical view points (x,y) to pixel
  # indices (a,b).  In particular, define s_x, o_x, s_y, and o_y so that:
  #  - For any pixel (a,b)...
  #      x = o_x + s_x * a
  #      y = o_y + s_y * b
  #    ... defines the (x,y) position of the center of that pixel.
  #  - Therefore, for any given (x,y)...
  #      a = round((x - o_x)/s_x)
  #      b = round((y - o_y)/s_y)
  #    ... defines the pixel that (x,y) falls in the interior of, which is
  #    equivalent to...
  #      a = int(0.5 + (x - o_x)/s_x)
  #      b = int(0.5 + (y - o_y)/s_y)
  # (Note that x = 1 produces A = W in this case, so view-space coordinates near
  # the right side or top of the view volume need to be handled carefully.)
  o_x = 1.0/W-1
  s_x = 2.0/W
  o_y = 1.0/H-1
  s_y = 2.0/H

  # Here we define some "lambdas" (simple functions) that encode the transforms
  # above.  Call them as, e.g., (a, b) = view2pix(x, y).
  view2pix = lambda x, y: (int(0.5 + (x - o_x)/s_x), int(0.5 + (y - o_y)/s_y))
  pix2view = lambda a, b: (o_x + s_x * a, o_y + s_y * b)

  # Find the bounding box of the triangle in view coordinates, and clip it
  # to the view volume.  Be careful about those view edges!
  min_x = max(-1, np.amin(verts[:,0]))
  max_x = min(1-1.0/W, np.amax(verts[:,0]))
  min_y = max(-1, np.amin(verts[:,1]))
  max_y = min(1-1.0/H, np.amax(verts[:,1]))

  # Convert to a bounding box in image coordinates, using view2pix.
  (min_a, min_b) = view2pix(min_x, min_y)
  (max_a, max_b) = view2pix(max_x, max_y)

  # For each (a, b) pair in this image-space bounding box:
  #  - Convert back to view coordinates.
  #  - Use pointOnTriangle() to get the view-space z-coordinate of this point
  #    on the triangle (if indeed it does fall on the triangle).
  #  - Use the z-coordinate and the z-buffer to decide whether to paint the
  #    color and rewrite the z-buffer at this pixel.
  for a in range(min_a, max_a+1):
    for b in range(min_b, max_b+1):
      (x, y) = pix2view(a, b)
      p = pointOnTriangle(x, y, verts)
      if not p == None:
        if p > z_buf[a,b] and p < 1:
          z_buf[a,b] = p
          img[a,b,:] = color

def pointOnTriangle(x, y, verts):
  """Returns the z-coordinate of a point on a triangle, given x and y.

  Arguments:
   - x: an x-coordinate in the canonical view volume.
   - y: a y-coordinate in the canonical view volume.
   - verts (3x4): the view-space (X,Y,Z,W) coordinates of a triangle's
     three vertices.  W is ignored and assumed to be 1.

  The behavior of this function depends on defining a line, parallel to the
  Z-axis, passing through all points (x,y,*,1).

  Returns:
   - None if the line does not intersect this triangle.
   - The z-coordinate of the intersection point if it exists.
  """
  # Compute barycentric coordinates.  For the three corners of the triangle
  # (call them v1, v2, and v3), we need a parameterization of points on the
  # triangle as
  #       v = beta * (alpha*v1 + (1-alpha)*v2) + (1-beta)*v3
  # Solving for alpha and beta is tedious, but the final result has a beautiful
  # symmetry to it:
  #
  #               | x  x2  x3 |
  #             - | y  y2  y3 |
  #               | 1   1   1 |
  # alpha = -------------------------
  #          | (x - x3)  (x1 - x2) |
  #          | (y - y3)  (y1 - y2) |
  #
  # beta = (x - x3)/(alpha * (x1 - x2) + (x2 - x3))
  #
  v = np.concatenate( ([[x, y]], verts[:,:2]), 0 )
  num = np.concatenate( (v[[0,2,3],:].T, np.ones((1,3))), 0 )
  den = np.array([[v[0,0] - v[3,0], v[1,0] - v[2,0]],
                  [v[0,1] - v[3,1], v[1,1] - v[2,1]]])
  a = -np.linalg.det(num) / np.linalg.det(den)
  b = (v[0,0]-v[3,0])/(a*(v[1,0]-v[2,0]) + (v[2,0]-v[3,0]))

  # Return None if this point is not on the triangle.  How can you tell
  # that from the Barycentric coordinates?
  if b*a < 0 or b*(1-a) < 0 or 1-b < 0:
    return None

  # Use the barycentric coordinates to interpolate the z-value, and
  # return it.
  return b*(a*verts[0,2]+ (1-a)*verts[1,2]) + (1-b)*verts[2,2]

def allTriData(instances, world_to_camera):
  """Returns vertex arrays, colors, and normals for all triangles.

  Arguments:
   - instances: a list of (inst_xform, node, surf) tuples, as produced
     by RootNode.getCompositeTransforms().
   - world_to_camera (4x4): transform from world to camera-centric
     coordinates, as returned by Camera.worldToCameraCentricXform().

  Returns (tri_verts, colors, normals).  To define these, say T is the
  total number of triangles used by all instances.
   - tri_verts (3xTx4): each tri_verts[i, t, :] is the homogeneous
     coordinates, in the camera-centric coordinate system, of vertex i of
     triangle t.
   - colors (Tx3): colors[t, :] is the surface color of triangle T.
   - normals (Tx3): normals[t, :] is the normal vector of triangle T, in
     the camera-centric coordinate system, with length equal to twice the
     triangle's area.
  """
  T = sum([node.mesh[1].shape[0] for (_b, node, _c) in instances])
  colors = np.zeros((T, 3))
  tri_verts = np.zeros((3, T, 4))
  t = 0
  for (inst_xform, node, surf) in instances:
    (verts, tris) = node.mesh
    T_here = tris.shape[0]
    colors[t:t+T_here] = surf.color
    # Transform to camera-centric coordinates.
    cam_verts = world_to_camera.dot(inst_xform).dot(verts)
    # Again, cam_verts[:, tris] uses tris as an array of indices into the vertex
    # array.  This gives us a 4xTx3 array, but we need 3xTx4, hence transpose.
    tri_verts[:, t:t+T_here, :] = cam_verts[:, tris].transpose((2,1,0))
    t += T_here
  # Get the normals of all the triangles.
  normals = triangleNormals(tri_verts)

  return (tri_verts, colors, normals)

def triangleNormals(X):
  """Returns the normal vectors of a set of triangles in space.

  The single argument, X, is a KxTxD numpy array, where K is at least 3,
  T is the number of triangles, and D is the dimensionality of the space
  in which we're working (2, 3, or 4).  For example:
    X[:,i,:] is the K vertices that describe triangle #i.
    X[2,i,0] is the X-coordinate of vertex #2 of triangle #i.
    X[2,i,1] is the Y-coordinate.
    X[2,i,2] is the Z-coordinate (if D >= 3).

  Since only three vertices are needed to describe a triangle, this
  routine only uses X[0,i,:], X[1,i,:], and X[2,i,:] for each i.  If
  K > 3, the higher-index slices of X are simply ignored.

  Also, if D = 4, we assume X[2,i,3] is the W-coordinate in homogeneous
  coords, and this value is ignored.  The caller is responsible for
  scaling the values to a common projection (same W value for all
  points), if required.

  The return value depends on D.
   - If D>=3, returns a Tx3 array: normal vectors for the T triangles.
   - If D=2, returns a 1-D array of length T.  For a particular triangle
     X[:,i,:], element i of this array is:
      * positive if the triangle is described in right-handed order
        (counterclockwise)
      * zero if the triangle is degenerate
      * negative if the triangle is described in a left-handed order
        (clockwise)
  """
  # Remove extraneous dimensions/values.
  X = X[:3, :, :3]
  return np.cross(X[1,:,:] - X[0,:,:], X[2,:,:] - X[0,:,:])

def axisNorm(X, axis=0):
  """Returns the norms of vectors in an array.

  Arguments:
   - X: a multidimensional numpy array.
   - axis (optional; default = 0): the axis along which to take the norm.

  Examples --- here we assume X is a 4-dimensional array, PxQxRxS.
   - axisNorm(X) returns a 1xQxRxS of the norms of every P-element vector.
   - axisNorm(X, 0) is equivalent to the above.
   - axisNorm(X, 2) returns a PxQx1xS.
  """
  return np.sqrt(np.sum(X**2, axis=axis, keepdims=True))
