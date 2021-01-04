import onnx
from functools import lru_cache
from mako.template import Template

apply = lambda f,*args,**kwargs:f(*args,**kwargs)

shape_to_c_array_proto = lru_cache()(lambda shape: ''.join(map('[{}]'.format, shape)))

get_layer_c_name = lru_cache()(lambda layer: 'layer_{}'.format(layer.replace('.', '_').replace('/', '_').replace(':', '_')))
get_node_c_name = lru_cache()(lambda layer: 'op_{}'.format(layer.replace('.', '_').replace('/', '_').replace(':', '_')))

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
  onnx.TensorProto.DataType.FLOAT: '-FLT_MAX',
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

conv2d_general_format = lambda node: {
  'name': get_node_c_name(node.name),
  'ctype': c_data_type[node.input[0].data_type],

  'i_ch': node.input[0].shape[1],
  'i_x': node.input[0].shape[2],
  'i_y': node.input[0].shape[3],

  'o_ch': node.output[0].shape[1],
  'o_x': node.output[0].shape[2],
  'o_y': node.output[0].shape[3],
  
  'pads0': node.attr.pads[0],
  'pads1': node.attr.pads[1],
  'pads2': node.attr.pads[2],
  'pads3': node.attr.pads[3],
  
  'strides_x': node.attr.strides[0],
  'strides_y': node.attr.strides[1],

  'bias': node.input[2].c_name + '[c]' if len(node.input) >= 3 else 0,
  'conv_l': node.input[1].shape[2],
  'weight': node.input[1].c_name,
}

conv2d_general = Template("""
// ${i_ch}x${i_x}x${i_y} => ${o_ch}x${o_x}x${o_y}
void ${name}(void* in, void* out)
{
  ${ctype} (*i)[${i_ch}][${i_x}][${i_y}];
  i = (${ctype} (*)[${i_ch}][${i_x}][${i_y}])(in);
  ${ctype} (*o)[${o_ch}][${o_x}][${o_y}];
  o = (${ctype} (*)[${o_ch}][${o_x}][${o_y}])(out);
  ${ctype} (*temp)[${i_ch}][${i_x}+${pads0}+${pads2}][${i_y}+${pads1}+${pads3}] = (typeof(temp))malloc(sizeof(*temp));
  memset(*temp, 0, sizeof(*temp));
  {
    for(int a=0;a<${i_ch};a++) {
      for(int b=${pads0};b<${i_x}+${pads0};b++) {
        memcpy((*temp)[a][b]+${pads1}, (*i)[a][b], ${i_y}*sizeof(${ctype}));
      }
    }
  }

  {
    ${ctype} *p = (${ctype}*)(out);
    for(int c=0;c < ${o_ch};c++) {
      int cnt=0;
      while(cnt++ < ${o_x}*${o_y}) {
        *p++ = ${bias};
      }
    }
  }

  for(int c=0;c<${o_ch};c++) {
    for(int x=0;x<${o_x};x++) {
      for(int y=0;y<${o_y};y++) {
        double sum = 0.0;
        for(int c_i=0;c_i < ${i_ch};c_i++) {
          for(int m=0;m < ${conv_l};m++) {
            for(int n=0;n < ${conv_l};n++) {
              sum += (*temp)[c_i][x*${strides_x}+m][y*${strides_y}+n] * ${weight}[c][c_i][m][n];
            }
          }
        }
        (*o)[c][x][y] += sum;
      }
    }
  }
  free(*temp);
}
""", strict_undefined=True)

class ConvGeneralImpl():
  @staticmethod
  def suitable(node):
    determines = [
      lambda: True, # node._attributes[3].ints[0] == 0,
    ]
    return all(map(apply, determines))

  @staticmethod
  def getOp(node):
    return conv2d_general.render(**conv2d_general_format(node))

  @staticmethod
  def getOpName(node):
    return '{}'.format(conv2d_general_format(node)['name'])
  
depthwise_conv2d_general_format = lambda node: {
  'name': get_node_c_name(node.name),
  'ctype': c_data_type[node.input[0].data_type],

  'i_ch': node.input[0].shape[1],
  'i_x': node.input[0].shape[2],
  'i_y': node.input[0].shape[3],

  'o_x': node.output[0].shape[2],
  'o_y': node.output[0].shape[3],
  
  'pads0': node.attr.pads[0],
  'pads1': node.attr.pads[1],
  'pads2': node.attr.pads[2],
  'pads3': node.attr.pads[3],
  
  'strides_x': node.attr.strides[0],
  'strides_y': node.attr.strides[1],

  'bias': node.input[2].c_name + '[c]' if len(node.input) >= 3 else 0,
  'conv_l': node.input[1].shape[2],
  'weight': node.input[1].c_name,
}
  
depthwise_conv2d_general = Template("""
// ${i_ch}x${i_x}x${i_y} => ${i_ch}x${o_x}x${o_y}
void ${name}(void* in, void* out)
{
  ${ctype} (*i)[${i_ch}][${i_x}][${i_y}];
  i = (${ctype} (*)[${i_ch}][${i_x}][${i_y}])(in);
  ${ctype} (*o)[${i_ch}][${o_x}][${o_y}];
  o = (${ctype} (*)[${i_ch}][${o_x}][${o_y}])(out);
  ${ctype} (*temp)[${i_ch}][${i_x}+${pads0}+${pads2}][${i_y}+${pads1}+${pads3}] = (typeof(temp))malloc(sizeof(*temp));
  memset(*temp, 0, sizeof(*temp));
  {
    for(int a=0;a<${i_ch};a++) {
      for(int b=${pads0};b<${i_x}+${pads0};b++) {
        memcpy((*temp)[a][b]+${pads1}, (*i)[a][b], ${i_y}*sizeof(${ctype}));
      }
    }
  }

  {
    ${ctype} *p = (${ctype}*)(out);
    for(int c=0;c < ${i_ch};c++) {
      int cnt=0;
      while(cnt++ < ${o_x}*${o_y}) {
        *p++ = ${bias};
      }
    }
  }

  for(int c_i=0;c_i < ${i_ch};c_i++) {
    for(int x=0;x<${o_x};x++) {
      for(int y=0;y<${o_y};y++) {
        double sum = 0;
        for(int m=0;m < ${conv_l};m++) {
          for(int n=0;n < ${conv_l};n++) {
            sum += (*temp)[c_i][x*${strides_x}+m][y*${strides_y}+n] * ${weight}[c_i][0][m][n];
          }
        }
        (*o)[c_i][x][y] += sum;
      }
    }
  }
  free(*temp);
}
""", strict_undefined=True)

class DepthwiseConvGeneralImpl():
  @staticmethod
  def suitable(node):
    determines = [
      lambda: node.attr.group==node.input[0].shape[1], # node._attributes[3].ints[0] == 0,
    ]
    return all(map(apply, determines))

  @staticmethod
  def getOp(node):
    return depthwise_conv2d_general.render(**depthwise_conv2d_general_format(node))

  @staticmethod
  def getOpName(node):
    return '{}'.format(depthwise_conv2d_general_format(node)['name'])

conv2d_format = lambda node: {
  'name': get_node_c_name(node.name),
  'ctype': c_data_type[node.input[0].data_type],

  'i_ch': node.input[0].shape[1],
  'i_x': node.input[0].shape[2],
  'i_y': node.input[0].shape[3],

  'o_ch': node.output[0].shape[1],
  'o_x': node.output[0].shape[2],
  'o_y': node.output[0].shape[3],

  'strides_x': node.attr.strides[0],
  'strides_y': node.attr.strides[1],

  'bias': node.input[2].c_name + '[c]' if len(node.input) >= 3 else 0,
  'conv_l': node.input[1].shape[2],
  'weight': node.input[1].c_name,
}

conv2d = Template("""
// ${i_ch}x${i_x}x${i_y} => ${o_ch}x${o_x}x${o_y}
void ${name}(void* in, void* out)
{
  ${ctype} (*i)[${i_ch}][${i_x}][${i_y}];
  i = (${ctype} (*)[${i_ch}][${i_x}][${i_y}])(in);
  ${ctype} (*o)[${o_ch}][${o_x}][${o_y}];
  o = (${ctype} (*)[${o_ch}][${o_x}][${o_y}])(out);

  {
    ${ctype} *p = (${ctype}*)(out);
    for(int c=0;c < ${o_ch};c++) {
      int cnt=0;
      while(cnt++ < ${o_x}*${o_y}) {
        *p++ = ${bias};
      }
    }
  }

  for(int c=0;c<${o_ch};c++) {
    for(int x=0;x<${o_x};x++) {
      for(int y=0;y<${o_y};y++) {
        double sum = 0.0;
        for(int c_i=0;c_i < ${i_ch};c_i++) {
          for(int m=0;m < ${conv_l};m++) {
            for(int n=0;n < ${conv_l};n++) {
             sum += (*i)[c_i][x*${strides_x}+m][y*${strides_y}+n] * ${weight}[c][c_i][m][n];
            }
          }
        }
        (*o)[c][x][y] += sum;
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
  'name': get_node_c_name(node.name),
  'ctype': c_data_type[node.input[0].data_type],

  'i_ch': node.input[0].shape[1],
  'i_x': node.input[0].shape[2],
  'i_y': node.input[0].shape[3],

  'o_ch': node.output[0].shape[1],
  'o_x': node.output[0].shape[2],
  'o_y': node.output[0].shape[3],
  
  'x_stride': node.attr.strides[0],
  'y_stride': node.attr.strides[1],

  'bias': node.input[2].c_name + '[c]' if len(node.input) >= 3 else 0,
  'conv_b': -(node.input[1].shape[2]//2),
  'conv_e': (node.input[1].shape[2]//2),
  'weight': node.input[1].c_name,
}

conv2d_padding = Template("""
// ${i_ch}x${i_x}x${i_y} => ${o_ch}x${o_x}x${o_y}
void ${name}(void* in, void* out)
{
  ${ctype} (*i)[${i_ch}][${i_x}][${i_y}];
  i = (${ctype} (*)[${i_ch}][${i_x}][${i_y}])(in);
  ${ctype} (*o)[${o_ch}][${o_x}][${o_y}];
  o = (${ctype} (*)[${o_ch}][${o_x}][${o_y}])(out);
  
  {
    ${ctype} *p = (${ctype}*)(out);
    for(int c=0;c < ${o_ch};c++) {
      int cnt=0;
      while(cnt++ < ${o_x}*${o_y}) {
        *p++ = ${bias};
      }
    }
  }
  
  for(int o_c=0;o_c < ${o_ch} ;o_c++) {
    for(int o_x=0;o_x < ${o_x};o_x += ${x_stride}) {
      for(int o_y=0;o_y < ${o_y};o_y += ${y_stride}) {
        double sum = 0.0;
        for(int c_i=0;c_i < ${i_ch};c_i++) {
          for(int m=${conv_b};m <= ${conv_e};m++) {
            for(int n=${conv_b};n <= ${conv_e};n++) {
              if(o_x+m<0)
                continue;
              if(o_y+n<0)
                continue;
              if(o_x+m >= ${o_x})
                continue;
              if(o_y+n >= ${o_y})
                continue;
              sum += (*i)[c_i][o_x+m][o_y+n] * ${weight}[o_c][c_i][m-(${conv_b})][n-(${conv_b})];
            } // o_y
          } // o_x
        } // o_c
        (*o)[o_c][o_x][o_y] += sum;
      } // n
    } // m
  } // c_i
}
""", strict_undefined=True)

class ConvGeneralPaddingImpl():
  @staticmethod
  def suitable(node):
    determines = [
      lambda: all(map(lambda x:x==1, node.attr.pads)),# node._attributes[3].ints[0] != 0,
    ]
    return all(map(apply, determines))

  @staticmethod
  def getOp(node):
    return conv2d_padding.render(**conv2d_padding_format(node))
  
  @staticmethod
  def getOpName(node):
    return '{}'.format(conv2d_padding_format(node)['name'])


depthwise_conv2d_padding_format = lambda node: {
  'name': get_node_c_name(node.name),
  'ctype': c_data_type[node.input[0].data_type],

  'i_ch': node.input[0].shape[1],
  'i_x': node.input[0].shape[2],
  'i_y': node.input[0].shape[3],

  'o_x': node.output[0].shape[2],
  'o_y': node.output[0].shape[3],

  'x_stride': node.attr.strides[0],
  'y_stride': node.attr.strides[1],

  'bias': node.input[2].c_name + '[c]' if len(node.input) >= 3 else 0,
  'conv_b': -(node.input[1].shape[2] // 2),
  'conv_e': (node.input[1].shape[2] // 2),
  'weight': node.input[1].c_name,
}

depthwise_conv2d_padding = Template("""
// ${i_ch}x${i_x}x${i_y} => ${i_ch}x${o_x}x${o_y}
void ${name}(void* in, void* out)
{
  ${ctype} (*i)[${i_ch}][${i_x}][${i_y}];
  i = (${ctype} (*)[${i_ch}][${i_x}][${i_y}])(in);
  ${ctype} (*o)[${i_ch}][${o_x}][${o_y}];
  o = (${ctype} (*)[${i_ch}][${o_x}][${o_y}])(out);

  {
    ${ctype} *p = (${ctype}*)(out);
    for(int c=0;c < ${i_ch};c++) {
      int cnt=0;
      while(cnt++ < ${o_x}*${o_y}) {
        *p++ = ${bias};
      }
    }
  }

  for(int c_i=0;c_i < ${i_ch};c_i++) {
    for(int o_x=0;o_x < ${o_x};o_x += ${x_stride}) {
      for(int o_y=0;o_y < ${o_y};o_y += ${y_stride}) {
        double sum = 0.0;
          for(int m=${conv_b};m <= ${conv_e};m++) {
            for(int n=${conv_b};n <= ${conv_e};n++) {
              if(o_x+m<0)
                continue;
              if(o_y+n<0)
                continue;
              if(o_x+m >= ${o_x})
                continue;
              if(o_y+n >= ${o_y})
                continue;
              sum += (*i)[c_i][o_x+m][o_y+n] * ${weight}[c_i][0][m-(${conv_b})][n-(${conv_b})];
            } // o_y
          } // o_x
        (*o)[c_i][o_x][o_y] += sum;
      } // n
    } // m
  } // c_i
}
""", strict_undefined=True)


class DepthwiseConvGeneralPaddingImpl():
  @staticmethod
  def suitable(node):
    determines = [
      lambda: all(map(lambda x: x ==1, node.attr.pads)) and node.attr.group==node.input[0].shape[1],
    ]
    return all(map(apply, determines))

  @staticmethod
  def getOp(node):
    return depthwise_conv2d_padding.render(**depthwise_conv2d_padding_format(node))

  @staticmethod
  def getOpName(node):
    return '{}'.format(depthwise_conv2d_padding_format(node)['name'])

depthwise_conv2d_format = lambda node: {
  'name': get_node_c_name(node.name),
  'ctype': c_data_type[node.input[0].data_type],

  'i_ch': node.input[0].shape[1],
  'i_x': node.input[0].shape[2],
  'i_y': node.input[0].shape[3],

  'o_x': node.output[0].shape[2],
  'o_y': node.output[0].shape[3],

  'strides_x': node.attr.strides[0],
  'strides_y': node.attr.strides[1],

  'bias': node.input[2].c_name + '[c]' if len(node.input) >= 3 else 0,
  'conv_l': node.input[1].shape[2],
  'weight': node.input[1].c_name,
}

depthwise_conv2d = Template("""
// ${i_ch}x${i_x}x${i_y} => ${i_ch}x${o_x}x${o_y}
void ${name}(void* in, void* out)
{
  ${ctype} (*i)[${i_ch}][${i_x}][${i_y}];
  i = (${ctype} (*)[${i_ch}][${i_x}][${i_y}])(in);
  ${ctype} (*o)[${i_ch}][${o_x}][${o_y}];
  o = (${ctype} (*)[${i_ch}][${o_x}][${o_y}])(out);

  {
    ${ctype} *p = (${ctype}*)(out);
    for(int c=0;c < ${i_ch};c++) {
      int cnt=0;
      while(cnt++ < ${o_x}*${o_y}) {
        *p++ = ${bias};
      }
    }
  }

  for(int c_i=0;c_i < ${i_ch};c_i++) {
    for(int x=0;x<${o_x};x++) {
      for(int y=0;y<${o_y};y++) {
        double sum = 0.0;
        for(int m=0;m < ${conv_l};m++) {
          for(int n=0;n < ${conv_l};n++) {
            sum += (*i)[c_i][x*${strides_x}+m][y*${strides_y}+n] * ${weight}[c_i][0][m][n];
          }
        }
        sum += (*o)[c_i][x][y];
      }
    }
  }
}
""", strict_undefined=True)

class DepthwiseConvGeneralNoPaddingImpl():
  @staticmethod
  def suitable(node):
    determines = [
      lambda: all(map(lambda x:x==0, node.attr.pads)) and node.attr.group==node.input[0].shape[1], # node._attributes[3].ints[0] == 0,
    ]
    return all(map(apply, determines))

  @staticmethod
  def getOp(node):
    return depthwise_conv2d.render(**depthwise_conv2d_format(node))

  @staticmethod
  def getOpName(node):
    return '{}'.format(depthwise_conv2d_format(node)['name'])
  
conv_optimizer = [
  DepthwiseConvGeneralNoPaddingImpl,
  DepthwiseConvGeneralPaddingImpl,
  ConvGeneralPaddingImpl,
  ConvGeneralNoPaddingImpl,
  DepthwiseConvGeneralImpl,
  ConvGeneralImpl,
]

gemm_format = lambda node:{
  'name': get_node_c_name(node.name),
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
  i = (${ctype} (*)[${i_len}])(in);
  ${ctype} (*o)[${o_len}];
  o = (${ctype} (*)[${o_len}])(out);

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
  'name': get_node_c_name(node.name),
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
  'name': get_node_c_name(node.name),
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
  'name': get_node_c_name(node.name),
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
  i = (${ctype} (*)[${ch}][${i_x}][${i_y}])(in);
  ${ctype} (*o)[${ch}][${o_x}][${o_y}];
  o = (${ctype} (*)[${ch}][${o_x}][${o_y}])(out);

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
  'name': get_node_c_name(node.name),
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
  'name': get_node_c_name(node.name),
  'ctype': c_data_type[node.input[0].data_type],
  'epsilon': node.attr.epsilon, # node._attributes[0].f,
  'ch': node.input[0].shape[1],
  'size': node.input[0].shape[2] * node.input[0].shape[3],
  'scale': node.input[1].c_name,
  'bias': node.input[2].c_name,
  'running_mean': node.input[3].c_name,
  'running_var': node.input[4].c_name,
}

bn = Template("""
void ${name}(void* in, void* out)
{
  ${ctype} (*i)[${ch}][${size}];
  i = (typeof(i))(in);

  for(int c=0;c < ${ch};c++) {
    ${ctype}* p = (typeof(p))((*i)[c]);
    int cnt = 0;
    double sum = 0;
    double qsum = 0;
    while(cnt < ${size}) {
      sum += (*i)[c][cnt];
      cnt++;
    }
    sum /= ${size};
    cnt=0;
    while(cnt < ${size}) {
      qsum += ((*i)[c][cnt]-sum)*((*i)[c][cnt]-sum);
      cnt++;
    }
    qsum /= ${size};
    cnt=0;
    sum *= 0.1;
    qsum *= 0.1;
    sum += ${running_mean}[c]*0.9;
    qsum += ${running_var}[c]*0.9;
    while(cnt < ${size}) {
      (*i)[c][cnt] = ${scale}[c] * ((*i)[c][cnt] - sum) / sqrtf(qsum + ${epsilon}) + ${bias}[c];
      cnt++;
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
  'name': get_node_c_name(node.name),
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
  i = (${ctype} (*)[${ch}][${i_x}][${i_y}])(in);
  ${ctype} (*o)[${ch}][${o_x}][${o_y}];
  o = (${ctype} (*)[${ch}][${o_x}][${o_y}])(out);

  for(int c=0;c<${ch};c++) {
    for(int x=0, o_i=0;x<${i_x}-${shape_x}+1;x+=${strides_x}) {
      for(int y=0, o_j=0;y<${i_y}-${shape_y}+1;y+=${strides_y}) {
        double result=0;
        float div=0;
        for(int m=0;m<${shape_x};m++) {
          for(int n=0;n<${shape_y};n++) {
              if((x+m)<${i_x} && (y+n)<${i_y}) {
                result += (*i)[c][x+m][y+n];
                div += 1;
              }
          }
        }
        (*o)[c][o_i][o_j] = result/div;
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
  'name': get_node_c_name(node.name),
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
  'name': get_node_c_name(node.name),
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
    if(${B}[a][0][0] == 0)
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