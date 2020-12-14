import onnx
from functools import lru_cache
from mako.template import Template

apply = lambda f,*args,**kwargs:f(*args,**kwargs)

shape_to_c_array_proto = lru_cache()(lambda shape: ''.join(map('[{}]'.format, shape)))

get_layer_c_name = lru_cache()(lambda layer: 'layer_{}'.format(layer.replace('.', '_').replace('/', '_').replace(':', '_')))

c_data_type = {
  onnx.TensorProto.DataType.FLOAT: 'float',
  onnx.TensorProto.DataType.UINT8: 'uint8_t',
  onnx.TensorProto.DataType.INT8: 'int8_t',
  onnx.TensorProto.DataType.INT16: 'int16_t',
  onnx.TensorProto.DataType.INT32: 'int32_t',
  onnx.TensorProto.DataType.INT64: 'int64_t',
  onnx.TensorProto.DataType.STRING: 'char*',
  onnx.TensorProto.DataType.BOOL: 'bool',
  onnx.TensorProto.DataType.FLOAT16: 'float',
  onnx.TensorProto.DataType.DOUBLE: 'double',
  onnx.TensorProto.DataType.UINT32: 'uint32_t',
  onnx.TensorProto.DataType.UINT64: 'uint64_t',
}

min_value = {
  onnx.TensorProto.DataType.FLOAT: '-FLT_MIN',
  onnx.TensorProto.DataType.UINT8: '0',
  onnx.TensorProto.DataType.INT8: 'SCHAR_MIN',
  onnx.TensorProto.DataType.INT16: 'SHRT_MIN',
  onnx.TensorProto.DataType.INT32: 'INT_MIN',
  onnx.TensorProto.DataType.FLOAT16: '-FLT_MAX',
  onnx.TensorProto.DataType.DOUBLE: '-DBL_MAX',
  onnx.TensorProto.DataType.UINT32: '0',
  onnx.TensorProto.DataType.UINT64: '0',
}

get_pointer = lambda tensor, buf: ''.format(
  ctype=c_data_type[tensor.data_type],
  name=get_layer_c_name(tensor.name), 
  dims=shape_to_c_array_proto(tensor.shape),
  buf='mem')

conv2d_format = lambda node: {
  'name': 'op_' + node.name,
  'ctype': c_data_type[node.input[0].data_type],

  'i_ch': node.input[0].shape[1],
  'i_x': node.input[0].shape[2],
  'i_y': node.input[0].shape[3],

  'o_ch': node.output[0].shape[1],
  'o_x': node.output[0].shape[2],
  'o_y': node.output[0].shape[3],

  'c_strides': node.attr.strides[0], # node._attributes[-1].ints[0],

  'bias': node.input[2].c_name if len(node.input) >= 3 else 0,
  'conv_l': node.input[1].shape[2],
  'weight': node.input[1].c_name,
}

conv2d = Template("""
// ${i_ch}x${i_x}x${i_y} => ${o_ch}x${o_x}x${o_y}
void ${name}(void* in, void* out)
{
  ${ctype} (*i)[${i_ch}][${i_x}][${i_y}];
  i = (typeof(i))(in);
  ${ctype} (*o)[${o_ch}][${o_x}][${o_y}];
  o = (typeof(o))(out);

  {
    ${ctype} *p = (${ctype}*)((*o));
    for(int c=0;c < ${o_ch};c++) {
      int cnt=0;
      while(cnt++ < ${o_x}*${o_y}) {
        *p++ = ${bias}[c];
      }
    }
  }

  for(int c_i=0;c_i < ${i_ch};c_i++) {
    for(int m=0;m < ${conv_l};m++) {
      for(int n=0;n < ${conv_l};n++) {
        for(int c=0;c<${o_ch};c++) {
          ${ctype} t = ${weight}[c][c_i][m][n];
          if(${ctype}_IS_ZERO(t))
            continue;
          for(int x=0;x<${o_x};x++) {
            for(int y=0;y<${o_y};y++) {
              (*o)[c][x][y] += (*i)[c_i][x+m][y+n] * t;
            }
          }
        }
      }
    }
  }
}
""", strict_undefined=True)

class ConvGeneralNoPaddingImpl():
  @staticmethod
  def suitable(node):
    determines = [
      lambda: all(map(lambda x:x==0, node.attr.pads)), # node._attributes[3].ints[0] == 0,
    ]
    return all(map(apply, determines))

  @staticmethod
  def getOp(node):
    return conv2d.render(**conv2d_format(node))

  @staticmethod
  def getOpName(node):
    return '{}'.format(conv2d_format(node)['name'])

conv2d_padding_format = lambda node:{
  'name': 'op_' + node.name,
  'ctype': c_data_type[node.input[0].data_type],

  'i_ch': node.input[0].shape[1],
  'i_x': node.input[0].shape[2],
  'i_y': node.input[0].shape[3],

  'o_ch': node.output[0].shape[1],
  'o_x': node.output[0].shape[2],
  'o_y': node.output[0].shape[3],

  'x_stride': node.attr.strides[0], # node._attributes[-1].ints[0],
  'y_stride': node.attr.strides[1], # node._attributes[-1].ints[1],

  'bias': node.input[2].c_name if len(node.input) >= 3 else 0,
  'conv_b': -(node.input[1].shape[2]//2),
  'conv_e': (node.input[1].shape[2]//2),
  'weight': node.input[1].c_name,
}

conv2d_padding = Template("""
// ${i_ch}x${i_x}x${i_y} => ${o_ch}x${o_x}x${o_y}
void ${name}(void* in, void* out)
{
  ${ctype} (*i)[${i_ch}][${i_x}][${i_y}];
  i = (typeof(i))(in);
  ${ctype} (*o)[${o_ch}][${o_x}][${o_y}];
  o = (typeof(o))(out);
  
  {
    ${ctype} *p = (${ctype}*)((*o));
    for(int c=0;c < ${o_ch};c++) {
      int cnt=0;
      while(cnt++ < ${o_x}*${o_y}) {
        *p++ = ${bias}[c];
      }
    }
  }
  
  for(int c_i=0;c_i < ${i_ch};c_i++) {
    for(int m=${conv_b};m <= ${conv_e};m++) {
      for(int n=${conv_b};n <= ${conv_e};n++) {
        for(int o_c=0;o_c < ${o_ch} ;o_c++) {
          ${ctype} t = ${weight}[o_c][c_i][m-(${conv_b})][n-(${conv_b})];
          if(${ctype}_IS_ZERO(t))
            continue;
          for(int o_x=(m>=0?0:-m);o_x < ${o_x} - (m<0?0:m) ;o_x += ${x_stride}) {
            for(int o_y=(n>=0?0:-n);o_y < ${o_y} - (n<0?0:n) ;o_y += ${y_stride}) {
                (*o)[o_c][o_x][o_y] += (*i)[c_i][o_x+m][o_y+n] * t;
            } // o_y
          } // o_x
        } // o_c
      } // n
    } // m
  } // c_i
}
""", strict_undefined=True)

class ConvGeneralPaddingImpl():
  @staticmethod
  def suitable(node):
    determines = [
      lambda: any(map(lambda x:x!=0, node.attr.pads)),# node._attributes[3].ints[0] != 0,
    ]
    return all(map(apply, determines))

  @staticmethod
  def getOp(node):
    return conv2d_padding.render(**conv2d_padding_format(node))
  
  @staticmethod
  def getOpName(node):
    return '{}'.format(conv2d_padding_format(node)['name'])

conv_optimizer = [
  ConvGeneralPaddingImpl,
  ConvGeneralNoPaddingImpl,
]

gemm_format = lambda node:{
  'name': 'op_' + node.name,
  'ctype': c_data_type[node.input[0].data_type],
  'o_len': node.output[0].shape[1],
  'i_len': node.input[0].shape[1],
  'weight': node.input[1].c_name,
  'bias': node.input[2].c_name,
}

gemm = Template("""
// ${i_len} => ${o_len}
void ${name}(void* in, void *out)
{
  ${ctype} (*i)[${i_len}];
  i = (typeof(i))(in);
  ${ctype} (*o)[${o_len}];
  o = (typeof(o))(out);

  for(int m=0;m<${o_len};m++) {
    ${ctype} sum = ${bias}[m];
    for(int n=0;n<${i_len};n++) {
      sum += (*i)[n] * ${weight}[m][n];
    }
    (*o)[m] = sum;
  }
}
""", strict_undefined=True)

class GemmGeneralImpl():
  @staticmethod
  def suitable(node):
    return True

  @staticmethod
  def getOp(node):
    return gemm.render(**gemm_format(node))
  
  @staticmethod
  def getOpName(node):
    return '{}'.format(gemm_format(node)['name'])

gemm_optimizer = [
  GemmGeneralImpl,
]

relu_format = lambda node:{
  'name': 'op_' + node.name,
  'ctype': c_data_type[node.input[0].data_type],
  'len': node.output[0].size,
}

relu = Template("""
void ${name}(void* in, void* out)
{
  ${ctype}* p = (${ctype}*)in;
  int i = 0;
  while(i++ < ${len}) {
    if(*p<0)*p=0;
    p++;
  }
}
""", strict_undefined=True)

class ReluImpl():
  @staticmethod
  def suitable(node):
    return True

  @staticmethod
  def getOp(node):
    return relu.render(**relu_format(node))
  
  @staticmethod
  def getOpName(node):
    return '{}'.format(relu_format(node)['name'])

leakyrelu_format = lambda node:{
  'name': 'op_' + node.name,
  'ctype': c_data_type[node.input[0].data_type],
  'len': node.output[0].size,
  'alpha': node.attr.alpha, # node._attributes[0].f,
}

leakyrelu = Template("""
void ${name}(void* in, void* out)
{
  ${ctype}* p = (${ctype}*)in;
  int i = 0;
  while(i++ < ${len}) {
    if(*p<0)*p *= ${alpha};
    p++;
  }
}
""", strict_undefined=True)

class LeakyReluImpl():
  @staticmethod
  def suitable(node):
    return True

  @staticmethod
  def getOp(node):
    return leakyrelu.render(**leakyrelu_format(node))
  
  @staticmethod
  def getOpName(node):
    return '{}'.format(leakyrelu_format(node)['name'])

maxpool_format = lambda node:{
  'name': 'op_' + node.name,
  'ctype': c_data_type[node.input[0].data_type],
  'ctype_min': min_value[node.input[0].data_type],
  'ch': node.input[0].shape[1],
  'i_x': node.input[0].shape[2],
  'i_y': node.input[0].shape[3],
  'o_x': node.output[0].shape[2],
  'o_y': node.output[0].shape[3],
  'shape_x': node.attr.kernel_shape[0], # node._attributes[0].ints[0],
  'shape_y': node.attr.kernel_shape[1], # node._attributes[0].ints[1],
  'strides_x': node.attr.strides[0], # node._attributes[2].ints[0] if len(node._attributes)>=3 else 0,
  'strides_y': node.attr.strides[1], # node._attributes[2].ints[1] if len(node._attributes)>=3 else 0,
}

maxpool = Template("""
void ${name}(void* in, void* out)
{
  ${ctype} (*i)[${ch}][${i_x}][${i_y}];
  i = (typeof(i))(in);
  ${ctype} (*o)[${ch}][${o_x}][${o_y}];
  o = (typeof(o))(out);

  for(int c=0;c<${ch};c++) {
    for(int x=0, o_i=0;x<${i_x};x+=${strides_x}) {
      for(int y=0, o_j=0;y<${i_y};y+=${strides_y}) {
        ${ctype} max=${ctype_min};
        for(int m=0;m<${shape_x};m++) {
          for(int n=0;n<${shape_y};n++) {
            if(max < (*i)[c][x+m][y+n]) {
              max = (*i)[c][x+m][y+n];
            }
          }
        }
        (*o)[c][o_i][o_j] = max;
        o_j++;
      }
      o_i++;
    }
  }
}
""", strict_undefined=True)

class MaxPoolImpl():
  @staticmethod
  def suitable(node):
    return True

  @staticmethod
  def getOp(node):
    return maxpool.render(**maxpool_format(node))
  
  @staticmethod
  def getOpName(node):
    return '{}'.format(maxpool_format(node)['name'])

clip_format = lambda node:{
  'name': 'op_' + node.name,
  'ctype': c_data_type[node.input[0].data_type],
  'size': node.input[0].size,
  'min': node.attr.min, # node._attributes[0].f,
  'max': node.attr.max, # node._attributes[1].f,
}

clip = Template("""
void ${name}(void* in, void* out)
{
  ${ctype}* i = (typeof(i))(in);

  int cnt=0;
  while(cnt++<${size}) {
    if(*i>${max}) {
      *i = ${max};
    }else if(*i < ${min}) {
      *i = ${min};
    }
    i++;
  }
}
""", strict_undefined=True)

class ClipImpl():
  @staticmethod
  def suitable(node):
    return True

  @staticmethod
  def getOp(node):
    return clip.render(**clip_format(node))
  
  @staticmethod
  def getOpName(node):
    return '{}'.format(clip_format(node)['name'])

bn_format = lambda node:{
  'name': 'op_' + node.name,
  'ctype': c_data_type[node.input[0].data_type],
  'epsilon': node.attr.epsilon, # node._attributes[0].f,
  'ch': node.input[0].shape[1],
  'size': node.input[0].shape[2] * node.input[0].shape[3],
  'scale': node.input[1].c_name,
  'bias': node.input[2].c_name,
  'mean': node.input[3].c_name,
  'var': node.input[4].c_name,
}

bn = Template("""
void ${name}(void* in, void* out)
{
  ${ctype} (*i)[${ch}][${size}];
  i = (typeof(i))(in);

  for(int c=0;c < ${ch};c++) {
    ${ctype}* p = (typeof(p))((*i)[c]);
    int cnt = 0;
    while(cnt++ < ${size}) {
      *p = ${scale}[c] * (*p - ${mean}[c]) / sqrtf(${var}[c]*${var}[c] + ${epsilon}[c]) + ${bias}[c];
    }
  }
}
""", strict_undefined=True)

class BnImpl():
  @staticmethod
  def suitable(node):
    return True

  @staticmethod
  def getOp(node):
    return bn.render(**bn_format(node))
  
  @staticmethod
  def getOpName(node):
    return '{}'.format(bn_format(node)['name'])

averagepool_format = lambda node:{
  'name': 'op_' + node.name,
  'ctype': c_data_type[node.input[0].data_type],
  'ctype_min': min_value[node.input[0].data_type],
  'ch': node.input[0].shape[1],
  'i_x': node.input[0].shape[2],
  'i_y': node.input[0].shape[3],
  'o_x': node.output[0].shape[2],
  'o_y': node.output[0].shape[3],
  'shape_x': node.attr.kernel_shape[0], # node._attributes[0].ints[0],
  'shape_y': node.attr.kernel_shape[1], # node._attributes[0].ints[1],
  'strides_x': node.attr.strides[0], # node._attributes[2].ints[0] if len(node._attributes)>=3 else 0,
  'strides_y': node.attr.strides[1], # node._attributes[2].ints[1] if len(node._attributes)>=3 else 0,
}

averagepool = Template("""
void ${name}(void* in, void* out)
{
  ${ctype} (*i)[${ch}][${i_x}][${i_y}];
  i = (typeof(i))(in);
  ${ctype} (*o)[${ch}][${o_x}][${o_y}];
  o = (typeof(o))(out);

  for(int c=0;c<${ch};c++) {
    for(int x=0, o_i=0;x<${i_x};x+=${strides_x}) {
      for(int y=0, o_j=0;y<${i_y};y+=${strides_y}) {
        ${ctype} result=0;
        for(int m=0;m<${shape_x};m++) {
          for(int n=0;n<${shape_y};n++) {
              result += (*i)[c][x+m][y+n];
          }
        }
        (*o)[c][o_i][o_j] = result/(${shape_x}*${shape_y});
        o_j++;
      }
      o_i++;
    }
  }
}
""", strict_undefined=True)

class AveragePoolImpl():
  @staticmethod
  def suitable(node):
    return True

  @staticmethod
  def getOp(node):
    return averagepool.render(**averagepool_format(node))
  
  @staticmethod
  def getOpName(node):
    return '{}'.format(averagepool_format(node)['name'])

mul_format = lambda node:{
  'name': 'op_' + node.name,
  'ctype': c_data_type[node.input[0].data_type],
  'size': node.input[0].size,
  'B': node.input[1].c_name,
}

mulOp = Template("""
void ${name}(void* in, void* out)
{
  ${ctype}* i = (typeof(i))(in);

  int cnt=0;
  while(cnt++<${size}) {
    *i++ *= ${B}[0];
  }
}
""", strict_undefined=True)

class MulImpl():
  @staticmethod
  def suitable(node):
    return True

  @staticmethod
  def getOp(node):
    return mulOp.render(**mul_format(node))
  
  @staticmethod
  def getOpName(node):
    return '{}'.format(mul_format(node)['name'])

add_format = lambda node:{
  'name': 'op_' + node.name,
  'ctype': c_data_type[node.input[1].data_type],
  'B': node.input[0].c_name,
  'ch': node.input[1].shape[1],
  'i_x': node.input[1].shape[2],
  'i_y': node.input[1].shape[3],
}

addOp = Template("""
void ${name}(void* in, void* out)
{
  ${ctype} (*i)[${ch}][${i_x}][${i_y}] = (typeof(i))(in);

  for(int a=0;a<${ch};a++) {
    if(${B}[ch][0][0] == 0)
      continue;
    for(int b=0;b<${i_x};b++) {
      for(int c=0;c<${i_y};c++) {
        (*i)[a][b][c] += ${B}[ch][0][0];
      }
    }
  }
}
""", strict_undefined=True)

class AddImpl():
  @staticmethod
  def suitable(node):
    return True

  @staticmethod
  def getOp(node):
    return addOp.render(**add_format(node))
  
  @staticmethod
  def getOpName(node):
    return '{}'.format(add_format(node)['name'])

def make_indent(code, tab=' ', indent=0):
  code = code.strip()
  return '\n'.join(map(lambda line: tab*indent, code.split('\n')))

class OpImpl():
  @staticmethod
  def getConv(cnode):
    c = next(e for e in conv_optimizer if e.suitable(cnode))
    return c.getOp(cnode)

  @staticmethod
  def getConvCaller(cnode):
    c = next(e for e in conv_optimizer if e.suitable(cnode))
    return c.getOpName(cnode)
  
  @staticmethod
  def getGemm(cnode):
    c = next(e for e in gemm_optimizer if e.suitable(cnode))
    return c.getOp(cnode)

  @staticmethod
  def getGemmCaller(cnode):
    c = next(e for e in gemm_optimizer if e.suitable(cnode))
    return c.getOpName(cnode)

  @staticmethod
  def getRelu(cnode):
    return ReluImpl.getOp(cnode)
  
  @staticmethod
  def getReluCaller(cnode):
    return ReluImpl.getOpName(cnode)

  @staticmethod
  def getLeakyRelu(cnode):
    return ReluImpl.getOp(cnode)
  
  @staticmethod
  def getLeakyReluCaller(cnode):
    return ReluImpl.getOpName(cnode)

  @staticmethod
  def getMaxPool(cnode):
    return MaxPoolImpl.getOp(cnode)
  
  @staticmethod
  def getMaxPoolCaller(cnode):
    return MaxPoolImpl.getOpName(cnode)

  @staticmethod
  def getClip(cnode):
    return ClipImpl.getOp(cnode)
  
  @staticmethod
  def getClipCaller(cnode):
    return ClipImpl.getOpName(cnode)

  @staticmethod
  def getAveragePool(cnode):
    return AveragePoolImpl.getOp(cnode)
  
  @staticmethod
  def getAveragePoolCaller(cnode):
    return AveragePoolImpl.getOpName(cnode)

  @staticmethod
  def getBN(cnode):
    return BnImpl.getOp(cnode)

  @staticmethod
  def getBnCaller(cnode):
    return BnImpl.getOpName(cnode)

  @staticmethod
  def getAdd(cnode):
    return AddImpl.getOp(cnode)
  
  @staticmethod
  def getAddCaller(cnode):
    return AddImpl.getOpName(cnode)
  
  @staticmethod
  def getMul(cnode):
    return MulImpl.getOp(cnode)

  @staticmethod
  def getMulCaller(cnode):
    return MulImpl.getOpName(cnode)
  
if __name__ == '__main__':
  from operators import Layer
  l = Layer('input', (1, 1, 28, 28), onnx.TensorProto.DataType.FLOAT)
  print(c_type_declare(l))