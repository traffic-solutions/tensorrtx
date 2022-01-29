import os
import cv2
import time
import ctypes
import threading
import numpy as np
from numpy import random
from configuration_files.crossings import *


class Result(ctypes.Structure):
    _fields_ = [('elem', ctypes.POINTER(ctypes.c_float)), ('size', ctypes.c_int)]


class COCOConfigTRT:
    # TODO: Make a class like this for our custom model
    class_name = ACCEPTABLE_CLASSES
    engine_path = YOLOV5_ENGINE_PATH
    execute_file = os.path.join(TENSORRTX_PATH, 'yolov5', 'build', 'yolov5')
    score_thr = YOLO_CONFIDENCE
    iou_thr = 0.5
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in class_name]


class YOLOV5TRT(object):
    def __init__(self, config):
        self.input_h = YOLO_SIZE
        self.input_w = YOLO_SIZE
        self.engine_file_path = config.engine_path
        self.excute_file = config.execute_file
        self.categories = config.class_name
        self.score_thr = config.score_thr
        self.iou_thr = config.iou_thr
        self.engine_strBuffer = ctypes.create_string_buffer(bytes(self.engine_file_path, 'utf-8'))
        self.detector = ctypes.cdll.LoadLibrary(self.excute_file)
        self.detector.detect_img.restype = ctypes.POINTER(Result)
        self.detector.initialize(self.engine_strBuffer)

    def detect_img(self, img):
        threading.Thread.__init__(self)
        conf_thr = ctypes.c_float(self.score_thr)
        nms_thr = ctypes.c_float(self.iou_thr)
        res = self.detector.detect_img(img.shape[0], img.shape[1], img.ctypes.data_as(ctypes.c_char_p), conf_thr,
                                       nms_thr)
        length = res.contents.size
        assert length % 6 == 0
        pred = np.array([res.contents.elem[i] for i in range(length)])
        pred = np.reshape(pred, (-1, 6))
        result_boxes = pred[:, :4] if len(pred) else np.array([])
        result_scores = pred[:, 4] if len(pred) else np.array([])
        result_classid = pred[:, 5] if len(pred) else np.array([])
        if not np.any(result_boxes):
            return [], [], []
        result_boxes = self.xywh2xyxy(img.shape[0], img.shape[1], result_boxes)
        result_boxes = result_boxes.astype(np.int)
        result_classid = result_classid.astype(np.int)
        return result_boxes.tolist(), result_scores.tolist(), result_classid.tolist()

    def destroy(self):
        self.detector.destroy()

    def xywh2xyxy(self, origin_h, origin_w, x):
        """
        description:    Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        param:
            origin_h:   height of original image
            origin_w:   width of original image
            x:          A boxes numpy, each row is a box [center_x, center_y, w, h]
        return:
            y:          A boxes numpy, each row is a box [x1, y1, x2, y2]
        """
        y = np.zeros_like(x)
        r_w = self.input_w / origin_w
        r_h = self.input_h / origin_h
        if r_h > r_w:
            y[:, 0] = x[:, 0] - x[:, 2] / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2 - (self.input_h - r_w * origin_h) / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2 - (self.input_h - r_w * origin_h) / 2
            y /= r_w
        else:
            y[:, 0] = x[:, 0] - x[:, 2] / 2 - (self.input_w - r_h * origin_w) / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2 - (self.input_w - r_h * origin_w) / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2
            y /= r_h

        return y

    def plot_one_box(self, x, img, color=None, label=None, line_thickness=None):
        # Plots one bounding box on image img
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    def draw_result(self, image, result):
        img = image.copy()
        result_boxes, result_scores, result_classid = result
        for j in range(len(result_boxes)):
            box = result_boxes[j]
            self.plot_one_box(
                box,
                img,
                label="{}:{:.2f}".format(self.categories[int(result_classid[j])], result_scores[j]),
            )
        return img


if __name__ == "__main__":
    detector = YOLOV5TRT(COCOConfigTRT)
    img = cv2.imread('samples/zidane.jpg')
    for i in range(10):
        t1 = time.time()
        boxes, scores, classes = detector.detect_img(img)
    img = detector.draw_result(img, (boxes, scores, classes))
    cv2.imwrite('output.jpg', img)
    detector.destroy()
