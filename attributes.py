import onnxruntime
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
    self._attrs = node._attributes
    self._pad_mode = getAttrByName(self._attrs, 'auto_pad') or 'NOTSET'
    self._dims = len(node.input[0].shape) - 2
    self._wh = node.input[0].shape[2:]
    t = getAttrByName(self._attrs, 'kernel_shape')
    self._kernel_shape = t.ints if t is not None else node.input[1].shape[2:]

  @property
  def kernel_shape(self):
      return self._kernel_shape

  @property
  def auto_pad(self):
    return self._pad_mode

  @property
  def padMode(self):
    return self._pad_mode

  @property
  def dilations(self):
    return getAttrByName(self._attrs, 'dilations') or [1]*self._dims

  @property
  def pads(self):
    _auto_pad = getAttrByName(self._attrs, 'auto_pad')
    if _auto_pad is None or _auto_pad == 'NOTSET':
      return getAttrByName(self._attrs, 'pads').ints or [0, 0]*self._dims
    elif _auto_pad == 'SAME_UPPER':
      # TODO: caculate padding values
      return [2]*self._dims + [0]*self._dims
    elif _auto_pad == 'SAME_LOWER ':
      return [0]*self._dims + [2]*self._dims
    elif _auto_pad == 'VALID ':
      return [0, 0]*self._dims

  @property
  def strides(self):
      _strides = getAttrByName(self._attrs, 'strides')
      return _strides.ints or [1]*self._dims

class LeakyReluAttr():
  def __init__(self, node):
    self.alpha = property(lambda _:getAttrByName(node._attributes, 'alpha').f or 0.01)

class MaxPoolAttr():
  def __init__(self, node):
    self._attrs = node._attributes
    self._pad_mode = getAttrByName(self._attrs, 'auto_pad') or 'NOTSET'
    self.ceil_mode = property(lambda _: getAttrByName(self._attrs, 'ceil_mode') or 0)
    self._dims = len(node.input[0].shape) - 2
    self._wh = node.input[0].shape[2:]
    t = getAttrByName(self._attrs, 'kernel_shape')
    self._kernel_shape = t.ints if t is not None else node.input[1].shape[2:]

  @property
  def kernel_shape(self):
      return self._kernel_shape

  @property
  def auto_pad(self):
    return self._pad_mode

  @property
  def padMode(self):
    return self._pad_mode

  @property
  def dilations(self):
    return getAttrByName(self._attrs, 'dilations') or [1]*self._dims

  @property
  def pads(self):
    _auto_pad = getAttrByName(self._attrs, 'auto_pad')
    if _auto_pad is None or _auto_pad == 'NOTSET':
      return getAttrByName(self._attrs) or [0, 0]*self._dims
    elif _auto_pad == 'SAME_UPPER':
      # TODO: caculate padding values
      return [2]*self._dims + [0]*self._dims
    elif _auto_pad == 'SAME_LOWER ':
      return [0]*self._dims + [2]*self._dims
    elif _auto_pad == 'VALID ':
      return [0, 0]*self._dims

  @property
  def strides(self):
      _strides = getAttrByName(self._attrs, 'strides')
      return _strides.ints or [1]*self._dims


class AveragePoolAttr():
  def __init__(self, node):
    self._attrs = node._attributes
    self._pad_mode = getAttrByName(self._attrs, 'auto_pad') or 'NOTSET'
    self._dims = len(node.input[0].shape) - 2
    self._wh = node.input[0].shape[2:]
    t = getAttrByName(self._attrs, 'kernel_shape')
    self._kernel_shape = t.ints if t is not None else node.input[1].shape[2:]

  @property
  def kernel_shape(self):
      return self._kernel_shape

  @property
  def auto_pad(self):
    return self._pad_mode

  @property
  def padMode(self):
    return self._pad_mode

  @property
  def dilations(self):
    return getAttrByName(self._attrs, 'dilations') or [1]*self._dims

  @property
  def pads(self):
    _auto_pad = getAttrByName(self._attrs, 'auto_pad')
    if _auto_pad is None or _auto_pad == 'NOTSET':
      return getAttrByName(self._attrs) or [0, 0]*self._dims
    elif _auto_pad == 'SAME_UPPER':
      # TODO: caculate padding values
      return [2]*self._dims + [0]*self._dims
    elif _auto_pad == 'SAME_LOWER ':
      return [0]*self._dims + [2]*self._dims
    elif _auto_pad == 'VALID ':
      return [0, 0]*self._dims

  @property
  def strides(self):
      _strides = getAttrByName(self._attrs, 'strides')
      return _strides.ints or [1]*self._dims

class BatchNormalizationAttr():
  def __init__(self, node):
    self.epsilon = property(lambda _:getAttrByName(node.attribute, 'epsilon') or 1e-5)
    self.momentum = property(lambda _:getAttrByName(node.attribute, 'momentum') or 0.9)

class PadAttr():
  def __init__(self, node):
    self.mode = property(lambda _:getAttrByName(node.attribute, 'mode') or 'constant')

class DefaultAttr():
  def __init__(self, node):
    pass

attr_classes = {
  'Conv': ConvAttr,
  'LeakyRelu': LeakyReluAttr,
  'MaxPool': MaxPoolAttr,
  'AveragePool': AveragePoolAttr,
  'BatchNormalization': BatchNormalizationAttr,
}
    
def makeAttr(node):
  if node.op_type in attr_classes:
    return attr_classes[node.op_type](node)
  else:
    return DefaultAttr(node)
      