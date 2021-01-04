import numpy as np
import onnx
import onnx.helper as helper


onnx_model = onnx.load("mobilenet_v1_1.0_224.onnx")
graph = onnx_model.graph
node  = graph.node

for i in range(len(node)):
  if node[i].output[0] == 'Conv__224:0':
    print(i)


