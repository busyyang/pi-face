# -*- coding:utf-8 -*-
import cv2
import os, threading, argparse
import numpy as np
from net.models import DetectModel, EncodingModel
import utils.utils as utils
from timeit import default_timer as timer
from PIL import Image, ImageFont, ImageDraw
from face_dataset import database_sqlite
from config import cfg


class FaceRecognisor:
    def __init__(self, use_tpu, use_encoding):
        self.detect_model = DetectModel(cfg.detect_model_path_tpu, cfg.detect_model_path, use_tpu)
        self.use_encoding = use_encoding
        # -----------------------------------------------#
        #   对数据库中的人脸进行编码
        #   known_face_encodings中存储的是编码后的人脸
        #   known_face_names为人脸的名字
        # -----------------------------------------------#
        self.known_face_encodings = []
        self.known_face_names = []
        if use_encoding:
            self.encoding_model = EncodingModel(cfg.encoding_model_path_tpu, cfg.encoding_model_path, use_tpu)

            data = database_sqlite.select_record(cfg.face_database_path)
            for row in data:
                self.known_face_names.append(row[0])
                self.known_face_encodings.append(row[3])

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
        outputs = self.detect_model.predict(draw_rgb)
        rectangles = utils.get_bbox(outputs, height, width, thr=0.2)
        # 没有人脸直接返回
        if len(rectangles) == 0:
            return [], draw_rgb

        rectangles = np.array(rectangles)
        rectangles[:, 0] = np.clip(rectangles[:, 0], 0, height)
        rectangles[:, 1] = np.clip(rectangles[:, 1], 0, width)
        rectangles[:, 2] = np.clip(rectangles[:, 2], 0, height)
        rectangles[:, 3] = np.clip(rectangles[:, 3], 0, width)

        return rectangles, draw_rgb

    def cal_encoding(self, rectangle, draw_rgb):
        """
        calculation the the encoding with 128 elements for faces
        :param rectangle: numpy array, the location of face, only one face required
        :param draw_rgb: cv::Mat, in R-G-B channel order
        :return:
            face_encoding: numpy array, encoding with 128 elements
        """
        crop_img = draw_rgb[int(rectangle[0]):int(rectangle[2]), int(rectangle[1]):int(rectangle[3])]
        crop_img = cv2.resize(crop_img, (160, 160))

        face_encoding = self.encoding_model.predict(crop_img)[0]
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
        if self.use_encoding:
            face_encodings = []
            for rectangle in rectangles:
                face_encoding = self.cal_encoding(rectangle, draw_rgb)
                face_encodings.append(face_encoding)

            face_names = []
            for face_encoding in face_encodings:
                # 取出一张脸并与数据库中所有的人脸进行对比，计算得分
                matches = utils.compare_faces(self.known_face_encodings, face_encoding,
                                              tolerance=cfg.detect_face_threshold)
                name = "Unknown"
                # 找出距离最近的人脸
                face_distances = utils.face_distance(self.known_face_encodings, face_encoding)
                # 取出这个最近人脸的评分
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
                face_names.append(name)
        else:
            name = ""
        # -----------------------------------------------#
        #   画框~!~
        # -----------------------------------------------#
        names = []
        font = ImageFont.truetype(font='font/msyh.ttc',
                                  size=np.floor(3e-2 * img.shape[1] + 0.5).astype('int32'))
        thickness = (img.shape[0] + img.shape[1]) // 300

        img = Image.fromarray(img)
        for (top, left, bottom, right) in rectangles:
            draw = ImageDraw.Draw(img)
            if name in self.known_face_names or name == "":
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
        self.fail_time = 0  # 记录获取画面失败的次数
        super(CameraBufferCleanerThread, self).__init__(name=name)
        self.start()

    def run(self):
        while self.running_status:
            ret, self.last_frame = self.camera.read()
            if not ret:
                self.fail_time += 1
            else:
                self.fail_time = 0
            if self.fail_time > 10:
                print('无法打开视频源......')
                self.running_status = False
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--help', '-h', action='help',
                        help='run face detection and recognition on raspberry pi.')
    parser.add_argument('--use_tpu', '-t', action='store_true',
                        help='use tpu device if you have.')
    parser.add_argument('--use_encoding', '-e', action='store_true',
                        help='use encoding part, bad performance if on Raspberry PI')
    args = parser.parse_args()

    fr = FaceRecognisor(args.use_tpu, args.use_encoding)
    camera = cv2.VideoCapture(cfg.video_source)
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
            if not cam_cleaner.running_status:
                break
    cam_cleaner.running_status = False
    cv2.destroyAllWindows()
