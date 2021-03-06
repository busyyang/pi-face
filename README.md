## 在RPI中实现人脸识别的代码

### 技术路线
1. 通过SSD模型定位人脸位置，将人脸数据转化为160\*160的方框。
2. 使用facenet模型，将人脸数据转化为128维的特征向量。
3. 对比特征向量实现人脸识别。
4. 使用_edgetpu.tflite模型需要使用到Google TPU设备（如Coral USB加速器），在树莓派+TPU的组合上，能进行到约14 FPS的速度。
5. 模型来源：检测人脸的SSD模型来自于[Coral.ai](https://coral.ai/models/all/)，facenet模型来自于[bubbliiiing/facenet-keras](https://github.com/bubbliiiing/facenet-keras/tree/main/model_data)，这个只是权重文件，需要将整个模型文件保存下来，通过`convert.py`转换为tflite模型，然后使用[Edge TPU Compiler](https://colab.research.google.com/github/google-coral/tutorials/blob/master/compile_for_edgetpu.ipynb)转化为TPU加速的模型文件。

### 安装
请参考[Coral.ai](https://coral.ai/software/#edgetpu-runtime)安装edgetpu_runtime, tflite_runtime, pycoral.

### 使用
1. 运行`python app.py -t -e`代码，通过PyQt录入人脸信息，信息保存在`face_dataset/face_database.db`数据库(sqlite3)中。参数`-t`表示使用tpu进行计算，需要插上一个TPU设备，如果没有的话，会报错。可尝试不带参数运行，但是FPS非常低。参数`-e`表示需要使用encoding模型。
2. 数据库文件可以通过sqlite Expert软件打开，FACE表中一共有4个字段。
    - NAME, AGE分别是名字和年龄，年龄可为空
    - IMAGE保存的是160\*160人脸数据的base64字符串
    - ENCODING保存的是128维的编码信息，为numpy ndarray数据格式
    
    ![database_record](./assert/images/database_record.png)
3. 在下方的输入框中输入视频源信息，使用默认摄像头输入`0`即可，使用IP摄像头参考`rtsp://admin:12345@192.168.1.123:8554/live`的样式配置。
4. 打开摄像头，有视屏流后截取图像，输入姓名和年龄信息插入数据库即可。

    ![data_insert](./assert/images/data_insert.png)
5. 数据库准备好以后，运行`python face_recongize.py -t -e`文件进行人脸识别体验。

    ![result](./assert/images/result.png)

6. 当只使用人脸检测模型，不人脸识别的话，可不带参数`-e`:
    ~~~
    python face_recongize.py -t
    ~~~

### Performence
在计算机上以及RPI上分别测试，
|OS|Desc.|only face_detect|detect and recognise|
|--|--|--|--|
|PC|i7-7700 with Coral TPU|130 FPS|31 FPS|
|RPI|-|10 FPS|1.8 FPS|


### issue
1. 在转换tflite模型的时候，由于keras, tensorflow以及h5py的版本不匹配问题，可能导致出问题，一下配置方案是测试可用的版本(测试过tensorflow==1.13.1在树莓派上加载tflite模型的时候报错)：
    |配置一||
    |--|--|
    |package|version|
    |Keras|2.3.1|
    |tensorflow|1.14.0|
    |h5py|2.9.0|

### 参考
1. https://github.com/bubbliiiing/keras-face-recognition
