"""
_scenegraph_base.py --- Private base classes for the scenegraph library.
By Jadrian Miles, January 2016

Please don't use this module directly; it's imported by the scenegraph module.
"""

import numpy as np

class Node(object):
  def __init__(self, name=''):
    self.name = name
    self._xform = np.eye(4)
    self._children = []
    # Technically we don't really need to keep track of parents, as all edges of
    # the DAG point away from the root, and all traversals implemented below
    # respect the orientation of those edges.  But it might be handy at some
    # point, and it lets us enforce restrictions on certain Node subclasses by
    # overloading _addParent() (like Root, which shouldn't have a parent).
    self._parents = []
  
  def addChild(self, c):
    self._addChild(c)
    c._addParent(self)
  
  def printTree(self):
    self._printTree("")
  
  def _printTree(self, prefix):
    if len(prefix) == 0:
      print self
    else:
      print prefix[:-3] + "+- " + str(self)
    N = len(self._children)
    if N == 0:
      print prefix
    else:
      print prefix + " |"
    i = 1
    for c in self._children:
      if i < N:
        c._printTree(prefix + " |  ")
      else:
        c._printTree(prefix + "    ")
      i += 1
  
  def _addChild(self, c):
    if len(self._children) > 0:
      msg = "This node (%s)\n\talready has a child (%s)." % (str(self), str(c))
      raise TypeError(msg)
    self._children = [c]
  
  def _addParent(self, p):
    self._parents.append(p)
  
  def _getSurface(self):
    return None
  
  def _traverse(self, M=np.eye(4), surf=None):
    """Traverses the scenegraph and returns transforms for all objects.
    
    Returns a list of 3-tuples.  In each tuple, the first element is a
    4x4 numpy array for the composite transform of a shape.  The second
    element is a ShapeNode object.
    
    The argument M is the composite transform between the root and this
    node.  "surf" is the SurfaceNode most recently encountered.
    """
    if self._getSurface() is not None:
      surf = self._getSurface()
    
    tuple_lists = [c._traverse(M.dot(self._xform), surf)
      for c in self._children]
    # tuple_lists is now a list of lists of tuples.  We need to concatenate all
    # the tuples together into one big list.  Rather than getting itertools
    # involved, I just used a weird list comprehension:
    # c.f. http://stackoverflow.com/q/952914
    return [T for tuples in tuple_lists for T in tuples]
  
  def __str__(self):
    if self.name is not None and len(self.name) > 0:
      return self.__class__.__name__ + " '" + self.name + "'"
    else:
      return self.__class__.__name__
  
  def __repr__(self):
    return str(self)
