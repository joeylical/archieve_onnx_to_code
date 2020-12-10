from functools import partial

def getAttrByName(attrs, name):
  for attr in attrs:
    if attr.name == name:
      return attr
  else:
    return None

def makeProperty(default, value):
  return property(lambda:value or default)

class ConvAttr():
  def __init__(self, node):
    self._attrs = node.attribute
    pad_func = partial(makeProperty, 0)
    self.pads = map(pad_func, getAttrByName(node, 'pads'))
    strides_func = partial(makeProperty, 0)
    self.strides = map(strides_func, getAttrByName(node, 'strides'))

attr_classes = {
  'Conv': ConvAttr,
}
    
def makeAttr(node):
  if node.op_type in attr_classes:
    return attr_classes[note.op_type]
  else:
    return None
      