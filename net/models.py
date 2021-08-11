# encoding: utf-8
import tflite_runtime.interpreter as tflite
import cv2
import numpy as np


class Net():
    def __init__(self, lite_model):
        interpreter = tflite.Interpreter(model_path=lite_model,
                                         experimental_delegates=[tflite.load_delegate('edgetpu.dll')])
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        self.interpreter = interpreter
        self.input_details = input_details
        self.output_details = output_details

    def predict(self, input):
        input = cv2.resize(input, (self.input_details[0]['shape'][1], self.input_details[0]['shape'][2]))
        if len(self.input_details[0]['shape']) - len(input.shape) == 1:
            input = np.expand_dims(input, axis=0)
        self.interpreter.set_tensor(self.input_details[0]['index'], input.astype('uint8'))
        self.interpreter.invoke()


class DetectModel(Net):
    def predict(self, input):
        super(DetectModel, self).predict(input)
        outputs = []  # bbox, class_id, score, count if ssd face model
        for out_node in self.output_details:
            output = self.interpreter.get_tensor(out_node['index'])
            outputs.append(output)
        return outputs


class EncodingModel(Net):
    def predict(self, input):
        super(EncodingModel, self).predict(input)
        # there is only one output node in encoding model
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        scale, zero_point = self.output_details[0]['quantization']
        output = (output.astype('float32') - zero_point) * scale
        return output
