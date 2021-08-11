import sys
from operator import itemgetter
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt


# -----------------------------#
#   将长方形调整为正方形
# -----------------------------#
def rect2square(rectangles):
    w = rectangles[:, 2] - rectangles[:, 0]
    h = rectangles[:, 3] - rectangles[:, 1]
    l = np.maximum(w, h).T
    rectangles[:, 0] = rectangles[:, 0] + w * 0.5 - l * 0.5
    rectangles[:, 1] = rectangles[:, 1] + h * 0.5 - l * 0.5
    rectangles[:, 2:4] = rectangles[:, 0:2] + np.repeat([l], 2, axis=0).T
    return rectangles


# ---------------------------------#
#   图片预处理
#   高斯归一化
# ---------------------------------#
def pre_process(x):
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0 / np.sqrt(size))
    y = (x - mean) / std_adj
    return y


# ---------------------------------#
#   l2标准化
# ---------------------------------#
def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output


# ---------------------------------#
#   计算128特征值
# ---------------------------------#
def calc_128_vec(model, img):
    face_img = pre_process(img)
    pre = model.predict(face_img)
    pre = l2_normalize(np.concatenate(pre))
    pre = np.reshape(pre, [128])
    return pre


# ---------------------------------#
#   计算人脸距离
# ---------------------------------#
def face_distance(face_encodings, face_to_compare):
    if len(face_encodings) == 0:
        return np.empty((0))
    return np.linalg.norm(face_encodings - face_to_compare, axis=1)


# ---------------------------------#
#   比较人脸
# ---------------------------------#
def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.6):
    dis = face_distance(known_face_encodings, face_encoding_to_check)
    return list(dis <= tolerance)


def get_bbox(output, h, w, thr=0.5):
    bboxes = []
    if output[-1][0] > 0:
        for i in range(int(output[-1][0])):
            if output[2][0, i] > thr:
                box = output[0][0, i, :]
                box = [int(h * box[0]), int(w * box[1]), int(h * box[2]), int(w * box[3])]
                bboxes.append(box)
            else:
                break
    return bboxes
