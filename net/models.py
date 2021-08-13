# encoding: utf-8
import tflite_runtime.interpreter as tflite
import cv2, platform
import numpy as np

EDGETPU_SHARED_LIB = {
    'Linux': 'libedgetpu.so.1',
    'Darwin': 'libedgetpu.1.dylib',
    'Windows': 'edgetpu.dll'
}[platform.system()]


class Net():
    def __init__(self, lite_model, use_tpu):
        """
        从文件创建一个tflite.Interpreter对象
        NOTICE: 需要继承该类重写predict函数，否则推理部分无输出
        2021-08-11  Jie Y.  Init
        Args:
            lite_model: str, the path and filename of tpu model
            use_tpu: str, use tpu device if you have
        """
        if use_tpu:
            try:
                interpreter = tflite.Interpreter(model_path=lite_model,
                                                 experimental_delegates=[tflite.load_delegate(EDGETPU_SHARED_LIB)])
            except Exception as e:
                print("Can't load `{}` model to tpu. Please try to run without tpu.".format(lite_model))
                exit(-2)
        else:
            interpreter = tflite.Interpreter(model_path=lite_model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        self.interpreter = interpreter
        self.input_details = input_details
        self.output_details = output_details

    def predict(self, input):
        """
        推理方法。该方法中执行了推理的内容，但是没有输出，输出部分在子类的中实现
        2021-08-11  Jie Y.  Init
        Args:
            input: numpy.array, 图像数据，该方法只处理一张图片。若需有batch_size需要在外面resize输入
                                并在内部resize_tensor_input()

        Returns: None

        """
        input = cv2.resize(input, (self.input_details[0]['shape'][1], self.input_details[0]['shape'][2]))
        if len(self.input_details[0]['shape']) - len(input.shape) == 1:
            input = np.expand_dims(input, axis=0)
        self.interpreter.set_tensor(self.input_details[0]['index'], input.astype('uint8'))
        self.interpreter.invoke()


class DetectModel(Net):
    def predict(self, input):
        """
        重写推理方法。完成推理后，获得数据直接输出list
        2021-08-11  Jie Y.  Init
        Args:
            input: numpy.array, 图像数据

        Returns:
            outputs: list, 模型所有输出节点的输出数据

        """
        super(DetectModel, self).predict(input)
        outputs = []  # bbox, class_id, score, count if ssd face model
        for out_node in self.output_details:
            output = self.interpreter.get_tensor(out_node['index'])
            outputs.append(output)
        return outputs


class EncodingModel(Net):
    def predict(self, input):
        """
        重写推理方法。由于编码模型只有一个输出，并且要执行dequanzition将uint32的数据转化为float类型
        2021-08-11  Jie Y.  Init
        Args:
            input: numpy.array, 图像数据

        Returns:
            output: numpy.array, 编码模型的输出，转float后输出，数值范围为[-1, 1]

        """
        super(EncodingModel, self).predict(input)
        # there is only one output node in encoding model
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        scale, zero_point = self.output_details[0]['quantization']
        output = (output.astype('float32') - zero_point) * scale
        return output
