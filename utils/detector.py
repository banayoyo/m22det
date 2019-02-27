import os
import glob
import itertools
import cv2
import numpy as np
import tensorflow as tf
import argparse

from m2det import M2Det
from utils.generate_priors import generate_priors
from utils.nms import nms

class Detector:
    def __init__(self, model_path, input_size, num_classes, threshold):
        self.model_path = model_path
        self.input_size = input_size
        self.num_classes = num_classes
        self.threshold = threshold
        self.priors = generate_priors() #generate anchor box
        self.build()

    def build(self):
        self.inputs = tf.placeholder(tf.float32, [None, self.input_size, self.input_size, 3])
        self.net = M2Det(self.inputs, tf.constant(False), self.num_classes)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(self.sess, self.model_path)

    def decode_boxes(self, boxes):
        prior_width = self.priors[:, 2] - self.priors[:, 0]
        prior_height = self.priors[:, 3] - self.priors[:, 1]
        prior_center_x = 0.5 * (self.priors[:, 2] + self.priors[:, 0])
        prior_center_y = 0.5 * (self.priors[:, 3] + self.priors[:, 1])
        decode_bbox_center_x = boxes[:, 0] * 0.1 * prior_width # variance0
        decode_bbox_center_x += prior_center_x
        decode_bbox_center_y = boxes[:, 1] * 0.1 * prior_height # variance0
        decode_bbox_center_y += prior_center_y
        decode_bbox_width = np.exp(boxes[:, 2] * 0.2) # variance1
        decode_bbox_width *= prior_width
        decode_bbox_height = np.exp(boxes[:, 3] * 0.2) # variance1
        decode_bbox_height *= prior_height
        decode_bbox_xmin = decode_bbox_center_x - 0.5 * decode_bbox_width
        decode_bbox_ymin = decode_bbox_center_y - 0.5 * decode_bbox_height
        decode_bbox_xmax = decode_bbox_center_x + 0.5 * decode_bbox_width
        decode_bbox_ymax = decode_bbox_center_y + 0.5 * decode_bbox_height
        decode_bbox = np.concatenate((decode_bbox_xmin[:, None], decode_bbox_ymin[:, None],
                                      decode_bbox_xmax[:, None], decode_bbox_ymax[:, None]), axis=-1)
        decode_bbox = np.minimum(np.maximum(decode_bbox, 0.0), 1.0)
        return decode_bbox

    def preprocess(self, img):
        img = cv2.resize(img, (self.input_size, self.input_size), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img - 127.5) / 128.0
        return img

    def detect(self, img):
        img_h, img_w = img.shape[:2]
        img = self.preprocess(img)

        outs = self.sess.run(self.net.prediction, feed_dict={self.inputs: np.array([img])})[0]

        # shape of y_pred: (?, num_boxes, 4 + num_classes)
        boxes = outs[:, :4]
        preds = outs[:, 4:]
        decoded_boxes = self.decode_boxes(boxes)

        results = []
        for box, pred in zip(decoded_boxes, preds):
            xmin, ymin, xmax, ymax = box
            clsid = np.argmax(pred)
            if clsid == 0:
                # in the case of background
                continue
            clsid -= 1 # decrement to skip background class
            prob = np.max(pred)
            left = xmin * img_w
            top = ymin * img_h
            right = xmax * img_w
            bottom = ymax * img_h
            results.append([clsid, prob, left, top, right, bottom])

        if len(results) > 0:
            return nms(results, self.threshold)
        else:
            return {}
