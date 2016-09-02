"""
scenegraph --- A module full of node classes for building scenegraphs.
By Jadrian Miles, January 2016.

There are a handful of public node types defined in this module:
 - RootNode
 - ShapeNode
 - GroupNode
 - XformNode
 - TranslateNode
 - RotateNode
 - ScaleNode
 - ShearNode
 - SurfaceNode

There are also loads of private implementation details (classes, methods,
and attributes), each of which begins with an underscore, per Python
convention. (So the _scenegraph_base.Node class isn't anything you should
use, since that whole module is supposed to be private, and you shouldn't
call n._addParent() directly, since the method is private.)


== Common Methods ==
All node types have a few capabilities.  For a node called "n", you can do
the following:
  
  n.addChild(c)  # Add c as a child node of n.
  n.name         # Get the name of n.
  n.printTree()  # Print out the tree traversal of the scenegraph
                 #   "downstream" from n.  This includes duplicate branches
                 #   for each node that has multiple parents.

In addition, if you print out a node, evaluate it in the REPL, or pass it to
str(), then you'll get a nice-looking string representation.


== RootNode ==
Construct a root node like so:
  r = RootNode()

In addition, RootNode has a special public method:
  objs = r.getCompositeTransforms()

getCompositeTransforms() computes composed transformations for each object
in the scene.  Basically, it goes down every path that r.printTree() would
show you, multiplying together the transformation matrices until it reaches
the leaves.
  r.getCompositeTransforms() returns a list of 2-tuples, one for each object
in the scene.  In each tuple, the first element is a 4x4 numpy array for the
composite transform of a shape.  The second element is a reference to the
ShapeNode object to which that transform should be applied.
  Note that you shouldn't ever directly modify the ShapeNode object, nor
anything inside of it (like the vertices of the mesh).  Since you're dealing
with a *reference* to the ShapeNode, any changes you make will affect that
node, and anyone else who's looking at it (maybe via another reference) will
see the change.


== ShapeNode ==
The constructor for shape nodes takes two arguments:
 - mesh: a 2-tuple (verts, tris) where verts is a 4xN numpy array of vertex
   positions (in object space) and tris is a Tx3 numpy array of vertex
   indices for the triangles making up the surface mesh.  This is exactly
   the tuple returned by each shape function in the "meshes" module.
 - name: a string, giving this shape a name.

ShapeNode also makes the mesh a public attribute: for a shape node "s",
  s.mesh  # Get the mesh 2-tuple.


== GroupNode ==
The constructor for group nodes takes one argument, the name.  GroupNode is
the only node type that can have multiple children.


== Transformation Node Classes ==
There are five transformation node classes, and they're all very similar.
They differ essentially only in their constructors, the arguments to which
they pass on to the corresponding function in the "transforms" module in
order to create a transformation matrix.

  == XformNode ==
  An XformNode represents a general linear transform, represented by a
  matrix.  The constructor takes two arguments:
   - M: a 4x4 matrix (numpy array) representing the transform.
   - name: a string, giving this transform a name.
  
  == TranslateNode ==
  The constructor for TranslateNode takes two arguments:
   - vec: a 1-D, 3-element numpy array, the vector by which to translate.
   - name: a string, giving this transform a name.

  == RotateNode ==
  The constructor for RotateNode takes two arguments:
   - angles: a 1-D, 3-element numpy array, the yaw/pitch/roll angles.
   - name: a string, giving this transform a name.

  == ScaleNode ==
  The constructor for ScaleNode takes two arguments:
   - factors: a 1-D, 3-element numpy array, the X/Y/Z scale factors.
   - name: a string, giving this transform a name.

  == ShearNode ==
  The constructor for ShearNode takes four arguments:
   - shear_dim: the dimension in which the shear applies.
   - contrib_dim: the dimension that modulates the amount of shear.
   - factor (scalar): the scaling factor for the amount of shear.
   - name: a string, giving this transform a name.
  The dimension arguments should each be 0, 1, or 2, which indicate
  X, Y, or Z respectively.


== SurfaceNode ==
A surface node determines the surface characteristics of every descendent
below it, overriding any surface nodes above it.  Equivalently, on the path
between the root and a shape node, the last surface node determines the
surface characteristics of the shape instance resulting from this path.

A SurfaceNode has two public attributes, color and name, both of which
are passed into the constructor:
 - color (3-element 1-D numpy array, or list): red, green, and blue
   intensities, each in the range [0, 1].
 - name (string): a name for this node.  (Default in constructor: '')
"""

def test():
  # A little function to put the scenegraph structure through its paces.  We
  # construct the graph for a simple scene consisting of three cubes (one
  # rotated, one left alone at the origin, and one translated), and then print
  # out the tree traversal of the graph.  Here's what the scenegraph looks like:
  # 
  #           ,-> T_r -.
  #          |         V
  #     R -> G ------> S
  #          |         ^
  #           `-> T_t -'
  
  import numpy as np
  
  # We know we need a root node.
  r = RootNode()
  
  # We're going to have a group of three cubes.
  g = GroupNode("triplet")
  r.addChild(g)
  
  # Rather than None, we should really be passing in the result of a call to
  # meshes.cube() as the first argument.
  s = ShapeNode(None, "cube")
  
  t_1 = RotateNode(np.array([0.5, -1.0, 2.25]), "spin")
  t_2 = TranslateNode(np.array([1.75, 0.5, -2]), "scoot")
  
  # Rotate a copy of the square, and add it to the group.
  t_1.addChild(s)
  g.addChild(t_1)
  
  # Add the square to the group.
  g.addChild(s)
  
  # Translate a copy of the square, and add it to the group.
  t_2.addChild(s)
  g.addChild(t_2)
  
  # The printTree() method is defined for all nodes and prints out a tree view
  # of everything "downstream" from it.
  print "Here's what the scenegraph looks like, with duplicate parents"
  print "represented as whole duplicate branches..."
  print ""
  r.printTree()
  
  # Here's how we get all the composed transforms out of the scenegraph.
  objs = r.getCompositeTransforms()
  
  # The next step, of course, would be to loop through all the (mat, ShapeNode)
  # pairs in objs and apply the transformation matrix to the shape's vertex
  # array, producing a new array in the world frame that you'd then render into
  # the scene.
  #   Instead, we'll just print them all out nicely.
  print "r.getCompositeTransforms() finds %i objects in the scene:" % len(objs)
  i = 1
  for obj in objs:
    print ("%2i: " % i) + str(obj[1]) + " with transform:"
    print "    " + "\n    ".join([l for l in str(obj[0]).split('\n')])
    i += 1
  


# ------============= YOU DON'T NEED TO READ BELOW THIS LINE =============------
# ------============= GORY IMPLEMENTATION DETAILS BELOW!     =============------

import _scenegraph_base
import transforms

class ShapeNode(_scenegraph_base.Node):
  """A Shape node in a scenegraph.
  
  A ShapeNode has two public attributes, mesh and name, both of which are
  passed into the constructor:
   - mesh: a 2-tuple (verts, tris) where verts is a 4xN numpy array of
     vertex positions (in object space) and tris is a Tx3 numpy array of
     vertex indices for the triangles making up the surface mesh.
   - name: a string, giving this shape a name.
  """
  
  def __init__(self, mesh, name):
    self.mesh = mesh
    super(ShapeNode, self).__init__(name)
  
  def _addChild(self, c):
    msg = "Shape nodes can't have children."
    raise TypeError(msg)
  
  def _traverse(self, T, surf):
    return [(T, self, surf)]


class XformNode(_scenegraph_base.Node):
  def __init__(self, xform, name):
    super(XformNode, self).__init__(name)
    self._xform = xform

class TranslateNode(XformNode):
  def __init__(self, vec, name=''):
    super(TranslateNode, self).__init__(
      transforms.translate(vec),
      name + " <translate by %s>" % vec.flatten()
    )

class RotateNode(XformNode):
  def __init__(self, angles, name=''):
    super(RotateNode, self).__init__(
      transforms.rotate(angles),
      name + " <rotate by %s>" % str(tuple(angles.flatten()))
    )

class ScaleNode(XformNode):
  def __init__(self, factors, name=''):
    super(ScaleNode, self).__init__(
      transforms.scale(factors),
      name + " <scale by %s>" % str(tuple(factors.flatten()))
    )

class ShearNode(XformNode):
  def __init__(self, shear_dim, contrib_dim, factor, name=''):
    super(ShearNode, self).__init__(
      transforms.shear(shear_dim, contrib_dim, factor),
      name + " <shear in %s by %s*%s>" %
        ("XYZ"[shear_dim], factor, "XYZ"[contrib_dim])
    )


class GroupNode(_scenegraph_base.Node):
  def _addChild(self, n):
    if isinstance(n, RootNode):
      msg = "Root node can't be made a child of anything else."
      raise TypeError(msg)
    self._children.append(n)
    n._parents.append(self)


class SurfaceNode(_scenegraph_base.Node):
  """A Surface node in a scenegraph.
  
  A surface node determines the surface characteristics of every
  descendent below it, overriding any above.  Equivalently, on the path
  between the root and a shape node, the last surface node determines the
  surface characteristics of the shape instance resulting from this path.
  
  A SurfaceNode has two public attributes, color and name, both of which
  are passed into the constructor:
   - color (3-element 1-D numpy array, or list): red, green, and blue
     intensities, each in the range [0, 1].
   - name (string): a name for this node.  (Default in constructor: '')
  """
  
  def __init__(self, color, name=''):
    self.color = color
    super(SurfaceNode, self).__init__(name)
  
  def _getSurface(self):
    return self


class RootNode(_scenegraph_base.Node):
  def __init__(self):
    super(RootNode, self).__init__('')
  
  def getCompositeTransforms(self):
    """Traverses the scenegraph and returns transforms for all objects.
    
    Returns a list of 2-tuples.  In each tuple, the first element is a
    4x4 numpy array for the composite transform of a shape.  The second
    element is a ShapeNode object.
    """
    return self._traverse()
  
  def _addParent(self, p):
    msg = "Root node can't be made a child of anything else."
    raise TypeError(msg)


if __name__ == "__main__":
  test()
