#!/usr/bin/env python3

import os
import sys
import tensorflow as tf
from tensorflow.python.saved_model import signature_constants
from tensorflow.saved_model import build_tensor_info

tf_graph = tf.Graph()
session = tf.Session(graph = tf_graph)

model_name = sys.argv[1]
os.system("mkdir -p models")

with session.graph.as_default():
   saver = tf.train.import_meta_graph("graph_nn.meta")
   session.graph.finalize()
   saver.restore(session, "weights/%s"%model_name)

   name_to_node = dict()
   with open("data_spec_to_tf_names.txt") as f:
       for line in f:
           line = line.strip()
           a,b = line.split(" = ")
           name_to_node[a] = tf_graph.get_tensor_by_name(b)

   from test_input import test_input
   data = dict(
       (name_to_node[name], value)
       for (name, value) in test_input.items()
       if name != "labels"
   )
    
   logits = name_to_node["logits"]
   #print(logits)
   
   session.run(logits, data)
   
   inputs = {x:build_tensor_info(name_to_node[x]) for x in name_to_node if x != "logits" and x != "labels"}
   outputs = {x:build_tensor_info(name_to_node[x]) for x in name_to_node if x == "labels" or x == "logits"}
   
   detection_signature = (
      tf.saved_model.signature_def_utils.build_signature_def(
         inputs=inputs,
         outputs=outputs,
         method_name=signature_constants.PREDICT_METHOD_NAME))
   
   builder = tf.compat.v1.saved_model.builder.SavedModelBuilder('./models/%s' % model_name)
   builder.add_meta_graph_and_variables(
      session, [tf.saved_model.tag_constants.SERVING],
      signature_def_map={
         signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            detection_signature,
      },
      saver=saver,
   )
   builder.save()
   
