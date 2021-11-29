# encoding: utf-8
from face_dataset.face_database_monitor import Ui_MainWindow
from PyQt5 import QtCore, QtGui, uic, QtWidgets
from PyQt5.QtWidgets import QWidget, QPushButton, QApplication, QMainWindow, QGridLayout, QFileDialog, QMessageBox
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import sys, cv2, threading
from PIL import Image, ImageQt
from face_recognize import FaceRecognisor
from face_dataset import database_sqlite
import base64, argparse


class CameraBufferCleanerThread(threading.Thread):
    def __init__(self, camera, name='camera-buffer-clear-thread'):
        super(CameraBufferCleanerThread, self).__init__(name=name)

        self.camera = camera
        self.last_frame = None
        self.running_status = True
        self.start()

    def run(self):
        while self.running_status:
            ret, self.last_frame = self.camera.read()


class AppWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, use_tpu, use_encoding, parent=None):
        super(AppWindow, self).__init__(parent)
        self.setupUi(self)
        self.pbSaveImage.clicked.connect(self.saveImage)
        self.pbInsert.clicked.connect(self.insertDatabase)
        self.pbOpen.clicked.connect(self.open)
        self.save_mode = False
        self.fr = FaceRecognisor(use_tpu, use_encoding)

    def open(self):
        source = self.textEditSource.toPlainText()
        if source == '':
            QMessageBox.warning(self, 'Warning', '请填写视频源：\n若为本地摄像头填写0,1,2等数字即可\n'
                                                 '若为IP相机，请填写rtsp://地址')
            return
        try:
            source = int(source)
        except Exception as e:
            print('非本地相机，正在链接IP相机......')
        self.camera = cv2.VideoCapture(source)
        self.cap = CameraBufferCleanerThread(self.camera)
        self.display()

    def saveImage(self):
        self.save_mode = True

    def insertDatabase(self):
        rect, frame_rgb = self.fr.face_detect(self.cap.last_frame)
        if len(rect) == 0:
            # QMessageBox.warning(self, '错误', '未发现人脸')
            print('未发现人脸')
        elif len(rect) > 1:
            # QMessageBox.warning(self, '错误', '仅可录入一个人')
            print('仅可录入一个人')
        else:
            rect = rect[0]
            name = self.textEditName.toPlainText()
            age = self.textEdit_Age.toPlainText()
            try:
                age = int(age)
            except:
                age = ''
            crop_img = self.cap.last_frame[rect[0]:rect[2], rect[1]:rect[3]]
            crop_img = cv2.resize(crop_img, (160, 160))
            cv2.imencode('.jpg', crop_img)[1].tofile(f'./face_dataset/images/{name}.jpg')
            face_coding = self.fr.cal_encoding(rect, self.cap.last_frame)
            database_sqlite.insert_record('./face_dataset/face_database.db', {
                'NAME': name,
                'AGE': age,
                'IMAGE': base64.b64encode(open(f'./face_dataset/images/{name}.jpg', 'rb').read()),
                'ENCODING': face_coding
            })
            print('插入成功')
        self.save_mode = False
        self.display()

    def display(self):
        while True:
            if self.save_mode:
                break
            if self.cap.last_frame is not None:
                frame = self.cap.last_frame
                rect, frame_rgb = self.fr.face_detect(frame)
                if len(rect) > 0:
                    for (top, left, bottom, right) in rect:
                        color = (0, 255, 0)
                        cv2.rectangle(frame_rgb, (left, top), (right, bottom), color, 2)

                img = Image.fromarray(frame_rgb)
                d = ImageQt.ImageQt(img)  # 转化为Qt对象
                self.label.setPixmap(QPixmap.fromImage(d))
            cv2.waitKey(1)

    def closeEvent(self, event):
        self.cap.running_status = False
        event.accept()
        exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--help', '-h', action='help',
                        help='run face detection and recognition on raspberry pi.')
    parser.add_argument('--use_tpu', '-t', action='store_true',
                        help='use tpu device if you have.')
    parser.add_argument('--use_encoding', '-e', action='store_true',
                        help='use encoding part, bad performance if on Raspberry PI')
    args = parser.parse_args()

    app = QApplication(sys.argv)
    win = AppWindow(args.use_tpu, args.use_encoding)
    win.show()
    sys.exit(app.exec_())

    """
    IP相机设置样式
    rtsp://admin:12345@10.66.211.198:8554/live
    """
