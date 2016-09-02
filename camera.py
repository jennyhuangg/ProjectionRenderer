"""
HW5: Projection Renderer
Comp 630 W'16 - Computer Graphics
Phillips Academy
2015-1-25

By Jenny Huang
"""
import numpy as np
import math
from numpy.linalg import norm  # norm(v) is the length of vector v.
import transforms

class Camera(object):
  """An ideal perspective camera in a 3-D scene.
  """

  def __init__(self, eye, look_at, up, near, far,
               view_angle_h=45, view_angle_v=45):
    """Look-at / up-vector constructor for a Camera.

    Each point or vector argument is represented by either a 3-element
    list or a (3,) numpy array (a 1-D array with 3 elements).
    Arguments:
     - eye: Position of the camera.
     - look_at: Point in space that the camera is looking at.
     - up: The "up vector" for the camera.  This vector defines which
       direction is "upright" for the camera: the view of this vector
       would appear upright in the rendered image produced by this
       camera.  "up" does NOT have to be a unit vector.
     - near: the distance from the camera to the near clipping plane,
       in world-space units.  (near > 0)
     - far: the distance from the camera to the far clipping plane,
       in world-space units.  (far > 0)
     - view_angle_h: the angle, in degrees, between the left and right
       sides of the view volume.  The look vector bisects this angle.
       (Default is 45.)
     - view_angle_v: the angle, in degrees, between the bottom and top
       sides of the view volume.  The look vector bisects this angle.
       (Default is 45.)

    Note that, for a given eye point, there are an infinite number of
    "look-at" points that define the same view direction.  For a given
    view direction, there are an infinite number of "up" vectors that
    define the same camera orientation.

    Prerequisites:
     - look_at != eye.
     - up is not parallel (or antiparallel) to look_at - eye.
    """
    self.near = float(near)
    self.far = float(far)
    # The internally-stored view angles are in radians away from the
    # look vector.
    self.view_angle_h = math.pi*(view_angle_h/2.0)/180
    self.view_angle_v = math.pi*(view_angle_v/2.0)/180
    self.setPose(eye, look_at, up)

  def setPose(self, eye, look_at, up):
    """Set the pose of this camera with look-at & up.

    Each argument is represented by either a 3-element list or a
    (3,) numpy array (a 1-D array with 3 elements).  Arguments:
     - eye: Position of the camera.
     - look_at: Point in space that the camera is looking at.
     - up: The "up vector" for the camera.  This vector defines which
       direction is "upright" for the camera: the view of this vector
       would appear upright in the rendered image produced by this
       camera.  "up" does NOT have to be a unit vector.

    Note that, for a given eye point, there are an infinite number of
    "look-at" points that define the same view direction.  For a given
    view direction, there are an infinite number of "up" vectors that
    define the same camera orientation.

    Prerequisites:
     - look_at != eye.
     - up is not parallel (or antiparallel) to look_at - eye.
    """
    eye = np.asarray(eye)
    look_at = np.asarray(look_at)
    up = np.asarray(up)

    self.eye = eye
    self.z = (eye - look_at)/np.linalg.norm(eye - look_at)
    self.x = (np.cross(up, self.z))/np.linalg.norm(np.cross(up, self.z))
    self.y = np.cross(self.z, self.x)

  def setViewAngles(self, aspect_ratio, max_angle):
    """Sets the view angles, given a desired aspect ratio and max angle.

    Arguments:
     - aspect_ratio: a number, image width / image height.
     - max_angle: the maximum angle, in degrees, between opposite sides
       of the view volume.
    """
    a = math.pi*(max_angle/2.0)/180
    if aspect_ratio < 1:
      self.view_angle_v = a
      self.view_angle_h = math.asin(math.sin(a) * aspect_ratio)
    else:
      self.view_angle_h = a
      self.view_angle_v = math.asin(math.sin(a) / aspect_ratio)

  def naturalAspectRatio(self):
    """Returns the image aspect ratio implied by the view angles.

    The "natural" aspect ratio of the camera is the aspect ratio that
    an image should have in order for the view of the camera to be
    undistorted when rendered onto that image.
    """
    return math.sin(self.view_angle_h) / math.sin(self.view_angle_v)

  def worldToCameraCentricXform(self):
    """Returns the rigid transform aligning the world to the camera.

    Returns a 4x4 matrix (numpy array).  After applying this transform,
    the world will be translated and rotated so that the camera lies
    at the origin, with the world's axes aligned with the camera's.
    """
    return self.rotateAlignXform().dot(self.translateToOriginXform())

  def worldToCanonicalViewXform(self):
    """Returns the world-to-canonical-view transform for this camera.

    Returns a 4x4 matrix (numpy array).  After applying this transform,
    each point in the world that falls within this camera's view frustum
    will end up with the values (x/w), (y/w), (z/w) in the range [-1,1].
    """
    return self.perspectiveNormalizationXform().dot(self.worldToCameraCentricXform())

  def translateToOriginXform(self):
    """Returns the transform that translates this camera to the origin.

    Returns a 4x4 matrix (numpy array).
    """
    return np.array([[1, 0, 0, -self.eye[0]],
                     [0, 1, 0, -self.eye[1]],
                     [0, 0, 1, -self.eye[2]],
                     [0, 0, 0, 1]])

  def rotateAlignXform(self):
    """Returns the rotation that aligns the camera's axes to the world axes.

    Returns a 4x4 matrix (numpy array).
    """
    r = np.concatenate(([self.x], [self.y], [self.z]), 0)
    r = np.concatenate((r, np.array([[0,0,0]])), 0)
    r = np.concatenate((r, np.array([0,0,0,1]).reshape(-1,1)), 1)
    return r

  def perspectiveNormalizationXform(self):
    """Returns the perspective normalization transform for this camera.

    The perspective normalization transform acts on a world that has
    already been transformed so that the camera is in its standard pose
    (at the origin, looking along the -Z axis).  We also assume that
    all points in the world are in homogeneous coordinates with w=1.

    The result of this normalization is that each point in the world
    that falls within this camera's view frustum will end up with the
    values (x/w), (y/w), (z/w) in the range [-1,1].

    Returns a 4x4 matrix (numpy array).
    """
    return np.array([[1.0/np.tan(self.view_angle_h), 0, 0, 0],
                     [0, 1.0/np.tan(self.view_angle_v), 0, 0],
                     [0, 0, (self.far + self.near)/(self.far - self.near),
                      2*self.far*self.near/(self.far - self.near)],
                     [0, 0, -1, 0]])
