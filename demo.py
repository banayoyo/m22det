import os
import glob
import itertools
import cv2
import numpy as np
import tensorflow as tf
import argparse
import matplotlib.pyplot as plt

from utils.detector import Detector
from mscoco import table

def draw(frame, results):
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_type = cv2.LINE_AA
    text_color = (255, 255, 255)
    base_border_size = 4
    base_font_size = 0.8
    base_font_scale = 2
    ratio = max(frame.shape[:2]) / 1000
    border_size = int(base_border_size * ratio)
    font_size = float(base_font_size * ratio)
    font_scale = int(base_font_scale * ratio)

    for cls, result in results.items():
        result = result[::-1]
        for prob, coord in result:
            name, color = get_classes(cls)
            data = '{}: {}'.format(name, round(prob, 3))
            left, top, right, bottom = [int(i) for i in coord]
            (label_width, label_height), baseline = cv2.getTextSize(data, font, font_size, font_scale)
            cv2.rectangle(frame, (left, top), (right, bottom), color, border_size)
            cv2.rectangle(frame, (left, top), (left + label_width, top + label_height), color, -1)
            cv2.putText(frame, data, (left, top + label_height - border_size), font, font_size, text_color, font_scale, line_type)
            print('{}: {} - left: {}, top: {}, right: {}, bottom: {}'.format(name, prob, left, top, right, bottom))

def get_classes(index):
    obj = [v for k, v in table.mscoco2017.items()]
    sorted(obj, key=lambda x:x[0])
    classes = [j for i, j in obj]
    np.random.seed(420)
    colors = np.random.randint(0, 224, size=(len(classes), 3))
    return classes[index], tuple(colors[index].tolist())

def main(args):
    
    img =cv2.imread(args.inputs)
    b, g, r = cv2.split(img)
    img2 = cv2.merge([r,g,b])
    # img2 = img[:,:,::-1]    this can be faster
    plt.subplot(121);plt.imshow(img)  # expects dis-sorted color
    plt.subplot(122);plt.imshow(img2)  # expects true color
    plt.show()
    
    det = Detector(
        model_path=args.model_path, 
        input_size=args.input_size, 
        num_classes=args.num_classes, 
        threshold=args.threshold)

    if args.inputs.endswith('.mp4'):
        cap = cv2.VideoCapture(args.inputs)
        while True:
            ret, img = cap.read()
            if not ret: break
            results = det.detect(img)
            draw(img, results)
            cv2.imshow('', img)
            cv2.waitKey(1)
    elif os.path.isdir(args.inputs):
        paths = glob.glob(os.path.join(args.inputs, '*'))
        for path in paths:
            img = cv2.imread(path)
            results = det.detect(img)
            draw(img, results)
            cv2.imshow('', img)
            cv2.waitKey(0)
    else:
        img = cv2.imread(args.inputs)
        results = det.detect(img)
        draw(img, results)
        cv2.imshow('', img)
        cv2.waitKey(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--model_path', default='C:/users/admin/desktop/git_repo/tf/data_file/darknet/')
    parser.add_argument('--input_size', type=int, required=True)
    parser.add_argument('--num_classes', type=int, default=80)
    parser.add_argument('--threshold', type=float, default=0.60)
    parser.add_argument('--gpu', type=str, default='-1')
    os.environ['CUDA_VISIBLE_DEVICES'] = parser.parse_args().gpu
    
    main(parser.parse_args())
