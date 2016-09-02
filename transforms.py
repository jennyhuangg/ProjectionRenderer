import numpy as np

def translate(v):
  """Returns the 4x4 matrix for a translation.
  
  "v" is a 1-D, 3-element numpy array, the vector by which to translate.
  """
  mat = np.eye(4)
  mat[0:3,3] = v
  return mat

def rotate(angles):
  """Returns the 4x4 matrix for a 3-D rotation.
  
  "angles" is a 1-D, 3-element numpy array of the yaw/pitch/roll angles, in
  radians.  We consider these angles to define a rotation in the following
  way (intrinsic Z-X-Y angles):
    1. The object (or space) is first yawed around its Z axis.
    2. It is then pitched around the newly-yawed X axis.
    3. Finally, it is rolled around the yawed-and-pitched Y axis.
  """
  # Unpack the yaw, pitch, and roll angles.
  (ty, tp, tr) = angles
  # Yaw around Z is in the X-Y plane.
  yaw = np.array(
    [[np.cos(ty), -np.sin(ty), 0, 0],
     [np.sin(ty),  np.cos(ty), 0, 0],
     [         0,           0, 1, 0],
     [         0,           0, 0, 1]]);
  # Pitch around X is in the Y-Z plane.
  pitch = np.array(
    [[1,          0,           0, 0],
     [0, np.cos(tp), -np.sin(tp), 0],
     [0, np.sin(tp),  np.cos(tp), 0],
     [0,          0,           0, 1]]);
  # Roll around Y is in the (-X)-Z plane.
  roll = np.array(
    [[ np.cos(tr), 0, np.sin(tr), 0],
     [          0, 1,          0, 0],
     [-np.sin(tr), 0, np.cos(tr), 0],
     [          0, 0,          0, 1]]);
  # We apply them in *reverse* order because we want, e.g. yaw to affect the
  # rotation axes for both pitch and roll.
  return yaw.dot(pitch.dot(roll))

def scale(factors):
  """Returns the 4x4 matrix for a 3-D scaling transform.
  
  "factors" is a 1-D, 3-element numpy array, the X/Y/Z scale factors.
  """
  return np.diag(np.concatenate((factors, [1])))

def shear(shear_dim, contrib_dim, factor):
  """Returns the 4x4 matrix for a 3-D shearing transform.
  
  Arguments:
   - shear_dim: the dimension in which the shear applies.
   - contrib_dim: the dimension that modulates the amount of shear.
   - factor (scalar): the scaling factor for the amount of shear.
  The dimension arguments should each be 0, 1, or 2, which indicate
  X, Y, or Z respectively.
  """
  M = np.eye(4)
  M[shear_dim, contrib_dim] += factor;
  return M
