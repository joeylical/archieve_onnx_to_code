import onnx
from mako.template import Template


shape_to_c_array_proto = lambda shape: ''.join(map('[{}]'.format, shape))

get_layer_c_name = lambda layer: 'layer_{}'.format(layer.replace('.', '_'))

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
  onnx.TensorProto.DataType.FLOAT: 'FLT_MIN',
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

conv2d_format = lambda node, i, o:{
  'name': 'op_' + node.name,
  'i_channel': shape_to_c_array_proto(node.input[0].shape),
  'o_shape': shape_to_c_array_proto(node.output[0].shape),
#   'weight_c':  
}

conv2d_padding = Template("""
// {i_ch}x{i_x}x{i_y} => {o_ch}x{o_x}x{o_y}
void {name}(void* in, void* out)
{
  {ctype} (*i)[{i_ch}][{i_x}][{i_y}] = (typeof(i))(in);
  {ctype} (*o)[{o_ch}][{o_x}][{o_y}] = (typeof(o))(out);
  
  for(int c=0;c < {o_ch};c++) {
    {ctype} *p;
    int cnt=0;
    while(cnt++ < {o_ch}*{o_x}*{o_y}) {
      *p++ = {bias}[o_c];
    }
  }
  
  
  for(int c_i=0;c_i < {i_c};c_i++) {
    for(int m={conv_b};m <= {conv_e};m++) {
      for(int n={conv_b};n <= {conv_e};n++) {
  
        {ctype} t = {weight}[c][c_i][m-{conv_b}][n-{conv_b}];
        if(IS_ZERO(t))
          continue;
          
        for(int o_c=0;o_c < {o_ch} ;o_c += {c_strip}) {
          for(int o_x=(m>=0?0:-m);o_x < {o_x} - (m>=0?0:-m) ;o_x += {x_strip}) {
            for(int o_y=(n>=0?0:-n);o_y < {o_y} - (n>=0?0:-n) ;o_y += {y_strip}) {
                (*o)[o_c][o_x][o_y] += (*i)[c_i][o_x+m][o_y+n] * t;
            } // o_y
          } // o_x
          //TODO: padding area
        } // o_c
      } // n
    } // m
  } // c_i
}
""")

def make_indent(code, tab=' ', indent=0):
  code = code.strip()
  return '\n'.join(map(lambda line: tab*indent, code.split('\n')))

class OpImpl():
  @staticmethod
  def getConv(cnode):
    return conv_padding.format()
  
if __name__ == '__main__':
  from operators import Layer
  l = Layer('input', (1, 1, 28, 28), onnx.TensorProto.DataType.FLOAT)
  print(c_type_declare(l))