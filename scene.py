"""
HW5: Projection Renderer
Comp 630 W'16 - Computer Graphics
Phillips Academy
2015-1-25

By Jenny Huang
"""
import numpy as np
import scenegraph as sg
import meshes
import our_shapes as os
#from gfx_helper_script import *
import gfx_helper_script as helper
from camera import Camera
import projection_renderer as render
from gfx_helper_plotting import *

def main():

  # Create a scenegraph with desired values.
  scene = makeScene(1, 5, 5, 2,
                    np.array([np.pi/4, 0, 0]),
                    np.array([-np.pi/2, 0, 0]),
                    np.array([0, np.pi/2, np.pi/4]),
                    10, 2.5,
                    np.array([0, np.pi/10, 0]),
                    np.array([0, -np.pi/5, 0]))

  # Create camera.
  eye = [13.2, -41.2, 19.0]
  look_at = [0, 0, 2.5]
  up = [0, 0, 1]
  im_width = 300
  im_height = 200
  camera = Camera(eye, look_at, up, 0.01, 300)
  camera.setViewAngles(float(im_width)/im_height, 35)

  # # Case 2.
  # eye = [-40.2, -13.2, 42.0]
  # look_at = [0, 0, 2.5]
  # up = [0, 0, 1]
  # im_width = 300
  # im_height = 300
  # camera = Camera(eye, look_at, up, 0.01, 300)
  # camera.setViewAngles(float(im_width)/im_height, 20)

  # # Case 3.
  # eye = [-30.2, -32.2, 19.0]
  # look_at = [0, 0, 2.5]
  # up = [0, 0, 1]
  # im_width = 350
  # im_height = 450
  # camera = Camera(eye, look_at, up, 83, 105)
  # camera.setViewAngles(float(im_width)/im_height, 34)

  # Wireframe.
  render.plotLines(scene, camera, im_width, im_height)

  # Rasterization.
  fig = helper.plt.figure()
  ax = fig.add_subplot(1, 1, 1, aspect='equal')
  img = render.renderRaster(scene, camera, im_width, im_height)
  drawImage(img, ax)
  helper.plt.show()

def makeScene(diameter, upper_arm, forearm, hand, shoulder, elbow, wrist,
              treeHeight, ballRadius, treeBend, trunkBend):
  """Returns the root node of the scenegraph for a scene that combines the
  robotic-arm scene and my custom scene.

  On the positive z side of the floor, there is a robotic arm. The arm is
  composed of three segments: the upper arm, forearm, and hand. Each of these
  has a square cross-section.

  On the negative z side of the floor, there is a squished ball. There are two
  trees, one on either side of the ball in the X-direction. Each tree is
  composed of one double cone (leaves and branches) on top of a cylinder
  (trunk). These all lie on a flat rectangular prism floor lying in the XY-plane
  with the origin of the world frame at the center of the top face.

  The eleven arguments are as follows:
   - diameter: the length of each side of the arm's square cross-section.
   - upper_arm, forearm, hand: the lengths of the segments.
   - shoulder, elbow, wrist: each one is a 1-D 3-element numpy array of
                             [yaw, pitch, roll] angles, in radians.
   - treeHeight: the height of each tree
   - ballRadius: the radius of the squishedBall
   - treeBend, trunkBend: each one is a 1-D 3-element numpy array of
                          [yaw, pitch, roll] angles, in radians. TreeBend
                          rotates both trees at the roots, and trunkBend
                          rotates both trees at the top of their trunks.

  Returns the root node of the scenegraph for this combination scene.
  """
  # Root node.
  r = sg.RootNode()

  # Rotate to correct orientation.
  rFinal = sg.RotateNode(np.array([0, np.pi/2.0, 0]), "final rotation")
  r.addChild(rFinal)

  # Group of four cubes.
  g_1 = sg.GroupNode("floor, upper arm, forearm, and hand group")
  rFinal.addChild(g_1)

  # Floor.
  t_2 = sg.TranslateNode(np.array([0, -0.5, 0]), "floor trans")
  g_1.addChild(t_2)
  s_1 = sg.ScaleNode(np.array([15,1,15]), "floor scale")
  t_2.addChild(s_1)
  su_floor = sg.SurfaceNode(np.array([21/256.0,112/256.0,39/256.0]), "f-surf")
  s_1.addChild(su_floor)

  # Group of upper arm, forearm, and hand.
  t_yay = sg.TranslateNode(np.array([0,0,3.5]), "whole arm translate")
  g_1.addChild(t_yay)
  r_1 = sg.RotateNode(shoulder, "shoulder rotation")
  t_yay.addChild(r_1)
  g_2 = sg.GroupNode("upper arm, forearm, and hand group")
  r_1.addChild(g_2)

  # Upper arm.
  s_2 = sg.ScaleNode(np.array([diameter, upper_arm, diameter]),
      "upper arm scale")
  g_2.addChild(s_2)
  su_uarm = sg.SurfaceNode(np.array([130, 23, 79])/256.0, "upper arm surf")
  s_2.addChild(su_uarm)

  # Group of forearm and hand.
  t_3 = sg.TranslateNode(np.array([0, upper_arm, 0]),
      "forearm and hand translation")
  g_2.addChild(t_3)
  r_2 = sg.RotateNode(elbow, "elbow rotation")
  t_3.addChild(r_2)
  g_3 = sg.GroupNode("forearm and hand group")
  r_2.addChild(g_3)

  # Forearm.
  s_3 = sg.ScaleNode(np.array([diameter,forearm, diameter]),
      "forearm scale")
  g_3.addChild(s_3)
  su_farm = sg.SurfaceNode(np.array([222, 27, 206])/256.0, "forearm surf")
  s_3.addChild(su_farm)

  # Hand.
  t_4 = sg.TranslateNode(np.array([0, forearm, 0]),
      "hand translation")
  g_3.addChild(t_4)
  r_3 = sg.RotateNode(wrist, "wrist rotation")
  t_4.addChild(r_3)
  s_4 = sg.ScaleNode(np.array([diameter, hand, diameter]),
      "hand scale")
  r_3.addChild(s_4)
  su_hand = sg.SurfaceNode(np.array([202, 151, 204])/256.0, "hand surf")
  s_4.addChild(su_hand)

  # First translation of arm up to origin.
  tFirst = sg.TranslateNode(np.array([0, 0.5, 0]), "arm translation")
  su_uarm.addChild(tFirst)
  su_farm.addChild(tFirst)
  su_hand.addChild(tFirst)

  # Scale cube down.
  scale = sg.ScaleNode(np.array([0.5,0.5,0.5]), "scale cube to unit cube")
  tFirst.addChild(scale)
  su_floor.addChild(scale)

  # Add cube shape node.
  c = sg.ShapeNode(meshes.cube(), "cube")
  scale.addChild(c)

  # My scene.

  # Group of squished ball and two trees.
  # One tree = one doubleCone + one cylinder
  tyay = sg.TranslateNode(np.array([0,0,-3.5]), "whole scene translate")
  g_1.addChild(tyay)
  g1 = sg.GroupNode("every shape group")
  tyay.addChild(g1)

  # SquishedBall.
  s2 = sg.ScaleNode(np.array([ballRadius, ballRadius, ballRadius]),
      "ball scale")
  g1.addChild(s2)
  t2 = sg.TranslateNode(np.array([0, 1, 0]), "above floor trans")
  s2.addChild(t2)
  su_ball = sg.SurfaceNode(np.array([91, 199, 252])/256.0, "squished ball surf")
  t2.addChild(su_ball)
  s = os.squishedBall(7)
  s = (os.vertsToHomogeneous(s[0]), s[1])
  squishedBall = sg.ShapeNode(s, "squished ball")
  su_ball.addChild(squishedBall)

  # Two trees.
  r2 = sg.RotateNode(treeBend, "trees rotate")
  g1.addChild(r2)

  # Group of two trees.
  g3 = sg.GroupNode("two trees")
  r2.addChild(g3)

  # Tree one.
  t3 = sg.TranslateNode(np.array([4, 0, 0]), "tree one trans")
  g3.addChild(t3)

  # Tree two.
  t4 = sg.TranslateNode(np.array([-4, 0, 0]), "tree one trans")
  g3.addChild(t4)

  # Group of tree one.
  g4 = sg.GroupNode("tree one group")
  t3.addChild(g4)

  # Group of tree two.
  g5 = sg.GroupNode("tree two group")
  t4.addChild(g5)

  # Two doubleCones.
  t5 = sg.TranslateNode(np.array([0, treeHeight/2.0, 0]))
  g4.addChild(t5)
  g5.addChild(t5)
  r3 = sg.RotateNode(trunkBend, "doubleCones rotate")
  t5.addChild(r3)
  s4 = sg.ScaleNode(np.array([1, 0.25*treeHeight, 1]), "doubleCones scale")
  r3.addChild(s4)
  t7 = sg.TranslateNode(np.array([0, 1, 0]), "above floor trans")
  s4.addChild(t7)
  su_cones = sg.SurfaceNode(np.array([41, 255, 66])/256.0, "doubleCones surf")
  t7.addChild(su_cones)
  d = os.doubleCone(12)
  d = (os.vertsToHomogeneous(d[0]), d[1])
  doubleCone = sg.ShapeNode(d, "double cone")
  su_cones.addChild(doubleCone)

  # Two cylinders.
  t6 = sg.ScaleNode(np.array([0.5, 0.25*treeHeight, 0.5]), "cylinders scale")
  g4.addChild(t6)
  g5.addChild(t6)
  t8 = sg.TranslateNode(np.array([0, 1, 0]), "above floor trans")
  t6.addChild(t8)
  su_cyls = sg.SurfaceNode(np.array([107, 71, 4])/256.0, "cylinders surface")
  t8.addChild(su_cyls)
  cylinder = sg.ShapeNode(meshes.prism(12), "cylinder")
  su_cyls.addChild(cylinder)

  return r

if __name__ == "__main__":
  main()
