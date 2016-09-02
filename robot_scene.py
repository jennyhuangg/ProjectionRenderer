import numpy as np
import scenegraph as sg
import meshes
from gfx_helper_mayavi import *

def main():
  # First, a bunch of simple arm poses, for the screenshots in the
  # assignment...
  #no_angles = np.array([0, 0, 0])
  #scene = makeScenegraph(1, 5, 5, 2, no_angles, no_angles, no_angles)
  #yaw_neg60 = np.array([-np.pi/3, 0, 0])
  #scene = makeScenegraph(1, 5, 5, 2, yaw_neg60, yaw_neg60, yaw_neg60)
  #pitch_60 = np.array([0, np.pi/3, 0])
  #scene = makeScenegraph(1, 5, 5, 2, pitch_60, pitch_60, pitch_60)
  #roll_30 = np.array([0, 0, np.pi/6])
  #scene = makeScenegraph(1, 5, 5, 2, roll_30, roll_30, roll_30)
  #scene = makeScenegraph(1, 5, 5, 2, np.array([0, 0, np.pi/4]),
  #        np.array([-np.pi/2, 0, 0]), np.array([0, np.pi/2, np.pi/4]))

  # A more exotic combination of joint angles
  shoulder = np.array([np.pi/10, 0, -np.pi/8])
  elbow = np.array([-np.pi/1.5, 0, np.pi/5])
  wrist = np.array([0, -np.pi/5, -np.pi/4])
  scene = makeScenegraph(1.5, 7, 5, 2, shoulder, elbow, wrist)

  #scene.printTree()

  instances = scene.getCompositeTransforms()

  fig = setUpFigure()

  # NB: the HW4 scenegraph code just gave (inst_xform, node) in each element of
  # "instances".  The updated scenegraph code for HW5 also includes surfaces,
  # which we must unpack even though we don't use them.
  for (inst_xform, node, surf) in instances:
    #print "======="
    #print inst_xform
    #print node
    (verts, tris) = node.mesh
    print inst_xform;
    drawTriMesh((inst_xform.dot(verts)), tris, fig)

  showFigure(fig)


def makeScenegraph(diameter, upper_arm, forearm, hand, shoulder, elbow, wrist):
  """Returns the root node of the scenegraph for a robotic-arm scene.

  The arm is composed of three segments: the upper arm, forearm, and hand.
  Each of these has a square cross-section.

  The seven arguments are as follows:
   - diameter: the length of each side of the arm's square cross-section.
   - upper_arm, forearm, hand: the lengths of the segments.
   - shoulder, elbow, wrist: each one is a 1-D 3-element numpy array of
                             [yaw, pitch, roll] angles, in radians.

  For more details, see the assignment.
  """
  s = sg.ShapeNode(meshes.cube(), "Cube")

  # Floor
  t_belowxy = sg.TranslateNode(np.array([0,0,-1]), "Put floor below X-Y plane")
  t_belowxy.addChild(s)
  t_floor = sg.ScaleNode(np.array([7.5, 7.5, 0.5]), "Resize floor")
  t_floor.addChild(t_belowxy)

  # Basic shape of hand, forearm, and upper arm: a diam/diam/1 box sitting on
  # top of the X-Y plane, so the hinge/joint is at the bottom of the box.
  #   The wrist rotation is expressed as yaw/pitch/roll, which are by default
  # around the Z/X/Y axes.  The assignment specifies that each block has its own
  # local coordinate system, in which Y goes along the long axis (global Z),
  # X comes out the side (same as global X), and Z comes out the back
  # (global -Y).
  #   So we will start all the boxes with their long axes along Y, and at the
  # end rotate up the whole assembly so it points along Z.
  t_hinge = sg.TranslateNode(np.array([0,1,0]), "Put the hinge on the bottom")
  t_hinge.addChild(s)
  t_diam = sg.ScaleNode(np.array([diameter/2.0, 0.5, diameter/2.0]),
    "Set diameter")
  t_diam.addChild(t_hinge)

  # Now the distinct shapes for hand, forearm, and upper arm.
  t_hand = sg.ScaleNode(np.array([1, hand, 1]), "Hand length")
  t_hand.addChild(t_diam)
  t_farm = sg.ScaleNode(np.array([1, forearm, 1]), "Forearm length")
  t_farm.addChild(t_diam)
  t_uarm = sg.ScaleNode(np.array([1, upper_arm, 1]), "Upper-arm length")
  t_uarm.addChild(t_diam)

  # Wrist, and positioning of hand at end of forearm.
  t_wrist = sg.RotateNode(wrist, "Wrist rotation")
  t_wrist.addChild(t_hand)
  t_hand_pos = sg.TranslateNode(np.array([0, forearm, 0]), "Hand -> end of arm")
  t_hand_pos.addChild(t_wrist)

  # Group the hand and forearm together into the "lower arm".
  g_lower = sg.GroupNode("Forearm and hand")
  g_lower.addChild(t_hand_pos)
  g_lower.addChild(t_farm)

  # Elbow, and positioning of the lower arm at the end of the upper arm.
  t_elbow = sg.RotateNode(elbow, "Elbow rotation")
  t_elbow.addChild(g_lower)
  t_arm_pos = sg.TranslateNode(np.array([0, upper_arm, 0]),
    "Forearm and hand -> end of upper arm")
  t_arm_pos.addChild(t_elbow)

  # Group the lower and upper arms together.
  g_arm = sg.GroupNode("Whole arm")
  g_arm.addChild(t_arm_pos)
  g_arm.addChild(t_uarm)

  # Shoulder joint at the origin.
  t_shoulder = sg.RotateNode(shoulder, "Shoulder rotation")
  t_shoulder.addChild(g_arm)

  # Finally, flip the arm up in place.  We need to rotate 90 degrees around
  # the global X axis, which is to say pitch by +pi/2.
  t_wholearm = sg.RotateNode(np.array([0, np.pi/2, 0]), "Put arm upright")
  t_wholearm.addChild(t_shoulder)

  # Make the root needs a group node as its child, for the floor and arm.
  g_scene = sg.GroupNode("Floor and arm")
  g_scene.addChild(t_floor)
  g_scene.addChild(t_wholearm)

  root = sg.RootNode()
  root.addChild(g_scene)

  return root


if __name__ == "__main__":
  main()
