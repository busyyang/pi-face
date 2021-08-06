"""
Convert Keras Model(.h5/.hdf5) to Tensorflow(.pb) or TF Lite(.tflite)
Since the custom_objects should be different by users,
please check the source code in your project, and comment `from main import custom_objects`
then load your custom_objects. If custom_objects is not in need, free free to set it as None.
if no custom_objects in main, it would be set to None as default.
tested on Win10x64, py3.7 with tensorflow 1.14 and keras 2.3.1
2020/4/29   Jie Y.      Init
usage:  python convert.py models/multi-task-pulsenet.h5 multi-task-pulsenet.tflite
    or: python convert.py models/multi-task-pulsenet.h5 multi-task-pulsenet.pb
"""

import tensorflow as tf
import os
import keras.backend as K
from keras.models import load_model
import argparse
import numpy as np

custom_objects = None

K.set_learning_phase(0)


def keras_to_tensorflow(keras_model, output_dir, model_name, out_prefix="output_", log_tensorboard=True):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    out_nodes = []

    for i in range(len(keras_model.outputs)):
        out_nodes.append(out_prefix + str(i + 1))
        tf.identity(keras_model.output[i], out_prefix + str(i + 1))

    sess = K.get_session()

    from tensorflow.python.framework import graph_util, graph_io

    init_graph = sess.graph.as_graph_def()

    main_graph = graph_util.convert_variables_to_constants(sess, init_graph, out_nodes)

    graph_io.write_graph(main_graph, output_dir, name=model_name, as_text=False)

    if log_tensorboard:
        from tensorflow.python.tools import import_pb_to_tensorboard

        import_pb_to_tensorboard.import_to_tensorboard(
            os.path.join(output_dir, model_name),
            output_dir)


def keras_to_tflite(keras_model, output_dir, des_model_name, custom_objects=None, test_mode=True):
    converter = tf.lite.TFLiteConverter.from_keras_model_file(keras_model, custom_objects=custom_objects)
    converter.post_training_quantize = True
    tflite_model = converter.convert()
    open(os.path.join(output_dir, des_model_name), "wb").write(tflite_model)
    if test_mode:
        interpreter = tf.lite.Interpreter(model_path=os.path.join(output_dir, des_model_name))
        interpreter.allocate_tensors()

        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        for i, _input in enumerate(input_details):
            input_shape = _input['shape']
            print('[{}]input size be:{}\n'.format(i, input_shape))
            input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
            interpreter.set_tensor(_input['index'], input_data)
        interpreter.invoke()

        for i, _output in enumerate(output_details):
            output_data = interpreter.get_tensor(_output['index'])
            print('[{}]the output is:{}'.format(i, output_data), '\twith size of', output_data.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('src_model_name', type=str, help='path and name of .h5 file')
    parser.add_argument('des_model_name', type=str, help='name of converted model file')

    args = parser.parse_args()

    assert args.src_model_name.split('.')[-1] in ['h5', 'hdf5'], 'src model should be keras model(.h5/.hdf5)'
    assert args.des_model_name.split('.')[-1] in ['tflite', 'pb'], 'des model should be TF Lite(.tflite) or TF(.pb)'

    if args.des_model_name.split('.')[-1] == 'tflite':
        output_dir = 'tflite_model'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        keras_to_tflite(args.src_model_name, output_dir, args.des_model_name, custom_objects)
    elif args.des_model_name.split('.')[-1] == 'pb':
        output_dir = 'tensorflow_model'
        keras_model = load_model(args.src_model_name, custom_objects=custom_objects)
        keras_model.summary()
        keras_to_tensorflow(keras_model, output_dir, args.des_model_name)