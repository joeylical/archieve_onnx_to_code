from functools import reduce
from operator import mul
import onnx
import numpy as np

class Layer():
  def __init__(self, name, shape, dType):
    self.name = name
    self.data_type = dType
    self.shape = shape
    self.c_name = 'layer_{}'.format(self.name.replace('.', '_'))
    
  def toArray(self):
    if self.data_type == onnx.TensorProto.DataType.FLOAT:
      declare = 'float {}{};'.format(self.c_name, ''.join(map('[{}]'.format, self.shape)))
      return declare
    elif self.data_type == 2:
      pass
    elif self.data_type == onnx.TensorProto.DataType.INT64:
      declare = 'uint64_t {}{};'.format(self.c_name, ''.join(map('[{}]'.format, self.shape)))
      return declare
    
  def __str__(self):
    return 'Tensor Object ' + self.name + ' ('+ ', '.join(map(str, self.shape)) + ')'
  
c_data_type = {
  onnx.TensorProto.DataType.FLOAT: 'float',
  onnx.TensorProto.DataType.UINT8: 'uint8_t',
  onnx.TensorProto.DataType.INT8: 'int8_t',
  onnx.TensorProto.DataType.INT16: 'int16_t',
  onnx.TensorProto.DataType.INT32: 'int32_t',
  onnx.TensorProto.DataType.STRING: 'char*',
  onnx.TensorProto.DataType.BOOL: 'bool',
  onnx.TensorProto.DataType.FLOAT16: 'float',
  onnx.TensorProto.DataType.DOUBLE: 'double',
  onnx.TensorProto.DataType.UINT32: 'uint32_t',
  onnx.TensorProto.DataType.UINT64: 'uint64_t',
}

shape_to_c_array_proto = lambda shape: ''.join(map(lambda d:'['+str(d)+']', shape))

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
    if self.data_type == onnx.TensorProto.DataType.FLOAT:
      declare = 'const float {}{} = \n'.format(self.c_name, ''.join(map('[{}]'.format, self.constant.shape)))
      body = array_to_c_code(self.constant, 0)
      return declare + body + ';'
    elif self.data_type == 2:
      pass
    elif self.data_type == onnx.TensorProto.DataType.INT64:
      declare = 'const uint64_t {}{} = \n'.format(self.c_name, ''.join(map('[{}]'.format, self.constant.shape)))
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
  
  def toCallSrc(self):
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
  def __init__(self, node):
    self.name = node.name
    self.input = node.input
    self.output = node.output
    self._attributes = node.attribute
    
  def toOpSrc(self):
    if self._attributes[3].ints[0] != 0:
      return """// {i.input[0].shape} -> {i.output[0].shape}
    void op_{i.name}()
    {{
      for(int c=0;c<{i.output[0].shape[1]};c++) {{
        for(int i=0;i<{i.output[0].shape[2]};i++) {{
          for(int j=0;j<{i.output[0].shape[3]};j++) {{
            {i.output[0].c_name}[0][c][i][j] = {i.input[2].c_name}[c];
            for(int c_i=0;c_i < {i.input[0].shape[1]};c_i++) {{
              for(int m=-{i.input[1].shape[3]}/2;m <= {i.input[1].shape[2]}/2;m++) {{
                for(int n=-{i.input[1].shape[2]}/2;n <= {i.input[1].shape[3]}/2;n++) {{
                  if(i+m < 0 || j+n < 0 || i+m>={i.input[0].shape[3]} || j+n>={i.output[0].shape[2]}) {{
                    // {i.output[0].c_name}[0][c][i][j] += {i.input[2].c_name}[c] * {i.input[1].c_name}[c][c_i][m+{i.input[1].shape[2]}/2][n+{i.input[1].shape[3]}/2];
                  }}else
                    {i.output[0].c_name}[0][c][i][j] += {i.input[0].c_name}[0][c_i][i+m][j+n] * {i.input[1].c_name}[c][c_i][m+{i.input[1].shape[2]}/2][n+{i.input[1].shape[3]}/2];
                }}
              }}
            }}
          }}
        }}
      }}
    }}
    """.format(i=self)
    else:
      return """// {i.input[0].shape} -> {i.output[0].shape}
    void op_{i.name}()
    {{
      for(int c=0;c<{i.output[0].shape[1]};c++) {{
        for(int i=0;i<{i.output[0].shape[2]};i++) {{
          for(int j=0;j<{i.output[0].shape[3]};j++) {{
            {i.output[0].c_name}[0][c][i][j] = {i.input[2].c_name}[c];
            for(int c_i=0;c_i < {i.input[0].shape[1]};c_i++) {{
              for(int m=0;m < {i.input[1].shape[2]};m++) {{
                for(int n=0;n < {i.input[1].shape[3]};n++) {{
                    {i.output[0].c_name}[0][c][i][j] += {i.input[0].c_name}[0][c_i][i+m][j+n] * {i.input[1].c_name}[c][c_i][m][n];
                }}
              }}
            }}
          }}
        }}
      }}
    }}
    """.format(i=self)
  
  def toCallSrc(self):
    return 'op_{i.name}();'.format(i=self)

class Gemm(CNode):
  inplace=False
  def __init__(self, node):
    self.name = node.name
    self.input = node.input
    self.output = node.output
    
  def toOpSrc(self):
    return """
    void op_{i.name}()
    {{
      for(int i=0;i<{i.output[0].shape[1]};i++) {{
        {ctype} sum = {i.input[2].c_name}[i];
        for(int j=0;j<{i.input[0].shape[1]};j++) {{
          sum += {i.input[0].c_name}[0][j] * {i.input[1].c_name}[i][j];
        }}
        {i.output[0].c_name}[0][i] = sum;
      }}
    }}
    """.format(i=self, ctype=c_data_type[self.input[0].data_type])
  
  def toCallSrc(self):
    return 'op_{i.name}();'.format(i=self) 

class Reshape(CNode):
  inplace=True
  def __init__(self, node):
    self.name = node.name
    self.input = node.input
    self.output = node.output
    
  def toOpSrc(self):
    return """
    void op_{i.name}()
    {{
      memcpy({i.output[0].c_name}, {i.input[0].c_name}, sizeof({i.input[0].c_name}));
    }}
    """.format(i=self)
  
  def toCallSrc(self):
    return 'op_{i.name}();'.format(i=self)

class Relu(CNode):
  inplace=True
  def __init__(self, node):
    self.name = node.name
    self.input = node.input
    self.output = node.output

  def toOpSrc(self):
    return """
    void op_{i.name}()
    {{
      {ctype}* p = ({ctype}*){i.input[0].c_name};
      int i = 0;
      while(i < {s}) {{
        if(*p<0)*p=0;
        i++;
        p++;
      }}
      memcpy({i.output[0].c_name}, {i.input[0].c_name}, sizeof({i.input[0].c_name}));
    }}
    """.format(i=self, ctype=c_data_type[self.input[0].data_type], s='*'.join(map(str, self.input[0].shape)))
  
  def toCallSrc(self):
    return 'op_{i.name}();'.format(i=self)
    
class MaxPool(CNode):
  inplace=False
  def __init__(self, node):
    self.name = node.name
    self.input = node.input
    self.output = node.output
    self.attribute = node.attribute

  def toOpSrc(self):
    return """
    void op_{i.name}()
    {{
      for(int c=0;c<{i.output[0].shape[1]};c++) {{
        for(int i=0, o_i=0;i<{i.output[0].shape[3]};i+={i.attribute[2].ints[0]}) {{
          for(int j=0, o_j=0;j<{i.output[0].shape[2]};j+={i.attribute[2].ints[1]}) {{
            {ctype} max=-9999;
            for(int m=0;m<{i.attribute[0].ints[0]};m++) {{
              for(int n=0;n<{i.attribute[0].ints[1]};n++) {{
                if(max < {i.input[0].c_name}[0][c][i+m][j+n]) {{
                  max = {i.input[0].c_name}[0][c][i+m][j+n];
                }}
              }}
            }}
            {i.output[0].c_name}[0][c][o_i][o_j] = max;
            o_j++;
          }}
          o_i++;
        }}
      }}
    }}
    """.format(i=self, ctype=c_data_type[self.input[0].data_type])
  
  def toCallSrc(self):
    return 'op_{i.name}();'.format(i=self)
    
class Constant(CNode):
  def __init__(self, node):
    self.name = node.name
    self.input = node.input
    self.output = node.output
  
  def toOpSrc(self):
    return ''
  
  def toCallSrc(self):
    return ''

class Softmax(CNode):
  inplace=True
  def __init__(self, node):
    self.name = node.name
    self.input = node.input
    self.output = node.output

  def toOpSrc(self):
    return """
    void op_{i.name}()
    {{
      memcpy({i.output[0].c_name}, {i.input[0].c_name}, sizeof({i.input[0].c_name}));
    }}
    """.format(i=self)
  
  def toCallSrc(self):
    return 'op_{i.name}();'.format(i=self)
    
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
