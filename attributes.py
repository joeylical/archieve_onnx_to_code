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

  @property
  def auto_pad(self):
    _auto_pad = getAttrByName(self._attrs, 'auto_pad')
    return __auto_pad or 

  @property
  def pads(self):
    _auto_pad = getAttrByName(self._attrs, 'auto_pad')
    if _auto_pad is None or _auto_pad == 'NOTSET':
      return getAttrByName(self._attrs) or [0, 0, 0, 0]
    elif _auto_pad == 'SAME_UPPER':
      return 
    elif _auto_pad == 'SAME_LOWER ':
      return
    elif _auto_pad == 'VALID ':
      return [0, 0, 0, 0]
attr_classes = {
  'Conv': ConvAttr,
}
    
def makeAttr(node):
  if node.op_type in attr_classes:
    return attr_classes[note.op_type]
  else:
    return None
      