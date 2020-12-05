from functools import reduce
from operator import mul
import onnx
import numpy as np
from op_impl import *

class Layer():
  def __init__(self, name, shape, dType):
    self.name = name
    self.data_type = dType
    self.shape = shape
    self.c_name = get_layer_c_name(self.name)
    
  def toArray(self):
    declare = '{} {}{};'.format(c_data_type[self.data_type], self.c_name, shape_to_c_array_proto(self.shape))
    return declare

  @property
  def size(self):
    return reduce(mul, self.shape)
    
  def __str__(self):
    return 'Tensor Object ' + self.name + ' ('+ ', '.join(map(str, self.shape)) + ')'

def array_to_c_code(array, indent):
  ind = '  ' * indent
  if isinstance(array[0], np.ndarray):
    result = ind + '{\n'
    r_list = []
    for subarr in array:
      r_list.append(array_to_c_code(subarr, indent+1))
    r_list.append(ind + '}')
    result += ',\n'.join(r_list)
    return result
  else:
    return ind + '{' + ', '.join(map(str, array)) + '}'

class ConstantLayer(Layer):
  def __init__(self, name, shape, dType, constant):
    super().__init__(name, shape, dType)
    self.constant = constant
    
  def toArray(self):
    declare = 'const {} {}{} = \n'.format(c_data_type[self.data_type], self.c_name, shape_to_c_array_proto(self.constant.shape))
    body = array_to_c_code(self.constant, 0)
    return declare + body + ';'
  
def getLayerByName(layers, name):
  for layer in layers:
    if layer.name == name:
      return layer
  else:
    return None

class CNode():
  name = 'base class'

  def update(self, layers):
    self.input = list(map(lambda i:getLayerByName(layers, i), self.input))
    self.output = list(map(lambda i:getLayerByName(layers, i), self.output))
  
  def toOpSrc(self):
    return '#error node op {} not implement'.format(self.name)
  
  def toCallSrc(self, i, o):
    return '#error node caller {} not implement'.format(self.name)
  
  @staticmethod
  def getOperator(node):
    if node.op_type in availible_nodes:
      return availible_nodes[node.op_type](node)
  
  def __str__(self):
    return 'cNode: {}'.format(self.name)

def getLayerByName(layers, name):
  for layer in layers:
    if layer.name == name:
      return layer
  else:
    return None
  
class Conv(CNode):
  inplace=False
  def __init__(self, node, relu=False):
    self.name = node.name
    self.input = node.input
    self.output = node.output
    self._attributes = node.attribute
    self.relu = relu
    
  def toOpSrc(self):
    return OpImpl.getConv(self)
  
  def toCallSrc(self, i, o):
    return '{name}({i}, {o});'.format(name=OpImpl.getConvCaller(self), i=i, o=o)

class Gemm(CNode):
  inplace=False
  def __init__(self, node):
    self.name = node.name
    self.input = node.input
    self.output = node.output
    
  def toOpSrc(self):
    return OpImpl.getGemm(self)

  def toCallSrc(self, i, o):
    return '{name}({i}, {o});'.format(name=OpImpl.getGemmCaller(self), i=i, o=o)

class Reshape(CNode):
  inplace=True
  def __init__(self, node):
    self.input = node.input
    self.output = node.output

  def toOpSrc(self):
    return ''
  
  def toCallSrc(self, i, o):
    return ''

class Relu(CNode):
  inplace=True
  def __init__(self, node):
    self.name = node.name
    self.input = node.input
    self.output = node.output

  def toOpSrc(self):
    return OpImpl.getRelu(self)

  def toCallSrc(self, i, o):
    return '{name}({i}, {o});'.format(name=OpImpl.getReluCaller(self), i=i, o=o)
    
class MaxPool(CNode):
  inplace=False
  def __init__(self, node):
    self.name = node.name
    self.input = node.input
    self.output = node.output
    self._attributes = node.attribute

  def toOpSrc(self):
    return OpImpl.getMaxPool(self)
  def toCallSrc(self, i, o):
    return '{name}({i}, {o});'.format(name=OpImpl.getMaxPoolCaller(self), i=i, o=o)
    
class Constant(CNode):
  inplace = True
  def __init__(self, node):
    self.name = node.name
    self.input = node.input
    self.output = node.output
  
  def toOpSrc(self):
    return ''
  
  def toCallSrc(self, i, o):
    return ''

class Softmax(CNode):
  inplace=True
  def __init__(self, node):
    self.name = node.name
    self.input = node.input
    self.output = node.output

  def toOpSrc(self):
    return ''
  
  def toCallSrc(self, i, o):
    return ''
    
availible_nodes = {
  'Conv': Conv,
  'Gemm': Gemm,
  'Reshape': Reshape,
  'Relu': Relu,
  'MaxPool': MaxPool,
  'Constant': Constant,
  'Softmax': Softmax
}

if __name__ == "__main__":
    print(availible_nodes)
