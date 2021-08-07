#-*- coding:utf-8 -*-
import cv2
import os, threading
import numpy as np
from net.mtcnn import mtcnn
import utils.utils as utils
from net.inception import InceptionResNetV1
from timeit import default_timer as timer
from PIL import Image, ImageFont, ImageDraw
from face_dataset import database_sqlite
import tensorflow as tf
import argparse


class FaceRecognisor:
    def __init__(self, use_lite_model=True):
        self.use_lite_model = use_lite_model

        # 创建mtcnn对象
        # 检测图片中的人脸
        self.mtcnn_model = mtcnn()
        # 门限函数
        self.threshold = [0.5, 0.8, 0.9]
        # 载入facenet
        # 将检测到的人脸转化为128维的向量
        if self.use_lite_model:
            print('Use Tflite model.')
            model_lite_path = './model_data/facenet.tflite'
            self.facenet_model = self.get_facenet_lite(model_lite_path)
        else:
            self.facenet_model = InceptionResNetV1()
            self.facenet_model.summary()
            model_path = './model_data/facenet_keras.h5'
            self.facenet_model.load_weights(model_path)

        # -----------------------------------------------#
        #   对数据库中的人脸进行编码
        #   known_face_encodings中存储的是编码后的人脸
        #   known_face_names为人脸的名字
        # -----------------------------------------------#
        self.known_face_encodings = []
        self.known_face_names = []
        data = database_sqlite.select_record('./face_dataset/face_database.db')
        for row in data:
            self.known_face_names.append(row[0])
            self.known_face_encodings.append(row[3])

    def get_facenet_lite(self, path):
        interpreter = tf.lite.Interpreter(model_path=path)
        interpreter.allocate_tensors()
        return interpreter

    def face_detect(self, draw):
        """
        detect face and return rectangles which is the location of faces
        2021-07-22  Jie Y.  add comments
        :param draw: cv::Mat, in B-G-R channel order
        :return:
            rectangles: numpy ndarray, which is the information of faces
            draw_rgb: cv::Mat, in R-G-B channel order
        """
        height, width, _ = np.shape(draw)
        draw_rgb = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

        # 检测人脸
        rectangles = self.mtcnn_model.detectFace(draw_rgb, self.threshold)
        # 没有人脸直接返回
        if len(rectangles) == 0:
            return rectangles, draw_rgb

        # 转化成正方形后再返回
        rectangles = utils.rect2square(np.array(rectangles, dtype=np.int32))
        rectangles[:, 0] = np.clip(rectangles[:, 0], 0, width)
        rectangles[:, 1] = np.clip(rectangles[:, 1], 0, height)
        rectangles[:, 2] = np.clip(rectangles[:, 2], 0, width)
        rectangles[:, 3] = np.clip(rectangles[:, 3], 0, height)
        return rectangles, draw_rgb

    def cal_encoding(self, rectangle, draw_rgb):
        """
        calculation the the encoding with 128 elements for faces
        :param rectangle: numpy array, the location of face, only one face required
        :param draw_rgb: cv::Mat, in R-G-B channel order
        :return:
            face_encoding: numpy array, encoding with 128 elements
        """
        landmark = (np.reshape(rectangle[5:15], (5, 2)) - np.array([int(rectangle[0]), int(rectangle[1])])) / (
                rectangle[3] - rectangle[1]) * 160

        crop_img = draw_rgb[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
        crop_img = cv2.resize(crop_img, (160, 160))

        new_img, _ = utils.Alignment_1(crop_img, landmark)
        new_img = np.expand_dims(new_img, 0)
        if self.use_lite_model:
            face_encoding = utils.calc_128_vec_lite(self.facenet_model, new_img)
        else:
            face_encoding = utils.calc_128_vec(self.facenet_model, new_img)
        return face_encoding

    def recognize(self, img):
        # -----------------------------------------------#
        #   人脸识别
        #   先定位，再进行数据库匹配
        # -----------------------------------------------#
        rectangles, draw_rgb = self.face_detect(img)
        if len(rectangles) == 0:
            return img, []
        # -----------------------------------------------#
        #   对检测到的人脸进行编码
        # -----------------------------------------------#
        face_encodings = []
        for rectangle in rectangles:
            face_encoding = self.cal_encoding(rectangle, draw_rgb)
            face_encodings.append(face_encoding)

        face_names = []
        for face_encoding in face_encodings:
            # 取出一张脸并与数据库中所有的人脸进行对比，计算得分
            matches = utils.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.9)
            name = "Unknown"
            # 找出距离最近的人脸
            face_distances = utils.face_distance(self.known_face_encodings, face_encoding)
            # 取出这个最近人脸的评分
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
            face_names.append(name)

        rectangles = rectangles[:, 0:4]
        # -----------------------------------------------#
        #   画框~!~
        # -----------------------------------------------#
        names = []
        font = ImageFont.truetype(font='font/msyh.ttc',
                                  size=np.floor(3e-2 * img.shape[1] + 0.5).astype('int32'))
        thickness = (img.shape[0] + img.shape[1]) // 300

        img = Image.fromarray(img)
        for (left, top, right, bottom), name in zip(rectangles, face_names):
            draw = ImageDraw.Draw(img)
            if name in self.known_face_names:
                color = (0, 255, 0)
                label = 'Name:{}'.format(name)
                names.append(name)
            else:
                color = (0, 0, 255)
                label = 'Unknown'

            text_origin = np.array([left + 2, bottom])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=color)
            draw.text(text_origin, label, fill=(0, 255, 0), font=font)
        return np.asarray(img), names


class CameraBufferCleanerThread(threading.Thread):
    def __init__(self, camera, name='camera-buffer-clear-thread'):
        """
        create a thread for process real-time video streaming
        :param camera: cv::VideoCapture object
        :param name: (optional) str, the name for thread
        """
        self.camera = camera
        self.last_frame = None
        self.running_status = True
        super(CameraBufferCleanerThread, self).__init__(name=name)
        self.start()

    def run(self):
        while self.running_status:
            ret, self.last_frame = self.camera.read()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_lite', '-l', action='store_true',
                        help='set this parameter if use tflite model')
    args = parser.parse_args()

    fr = FaceRecognisor(args.use_lite)
    camera = cv2.VideoCapture(0)
    cam_cleaner = CameraBufferCleanerThread(camera)
    while True:
        t1 = timer()
        if cam_cleaner.last_frame is not None:
            image = cam_cleaner.last_frame
            image, curnames = fr.recognize(image)
            t2 = timer()
            fps = t2 - t1
            cv2.putText(image, 'FPS:{:.1f}'.format(1 / fps), org=(3, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(255, 0, 0), thickness=2)
            try:
                cv2.imshow('Video', image)
            except Exception as e:
                continue
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cam_cleaner.running_status = False
    # camera.release()
    cv2.destroyAllWindows()
