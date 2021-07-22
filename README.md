## 在RPI中实现人脸识别的代码

### 技术路线
1. 通过MTCNN模型定位人脸位置，将人脸数据转化为160\*160的方框。
2. 使用facenet_keras.h5模型，将人脸数据转化为128维的特征向量。
3. 对比特征向量实现人脸识别。

### 使用
1. 通过`model_data/download_model.sh`下载`facenet_keras.h5`权重放到`model_data/`文件夹下。
2. 运行`app.py`代码，通过PyQt录入人脸信息，信息保存在`face_dataset/face_database.db`数据库(sqlite3)中。
    - 数据库中FACE表的字段有4个，NAME, AGE分别是名字和年龄，年龄可为空
    - IMAGE, ENCODING. 其中IMAGE保存的是160\*160人脸数据的base64字符串
    - ENCODING保存的是128维的编码信息，为numpy ndarray数据格式
3. 在下方的输入框中输入视频源信息，使用默认摄像头输入`0`即可，使用IP摄像头参考`rtsp://admin:12345@192.168.1.123:8554/live`的样式配置。
4. 打开摄像头，有视屏流后截取图像，输入姓名和年龄信息插入数据库即可。
5. 数据库准备好以后，打开`face_recongize.py`文件进行人脸识别体验。

### 参考
1. https://github.com/bubbliiiing/keras-face-recognition