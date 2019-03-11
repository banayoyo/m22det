import os
import cv2
import numpy as np
from torch.multiprocessing import Pool
from utils.nms_wrapper import nms
from utils.timer import Timer
from configs.CC import Config
import argparse
from layers.functions import Detect, PriorBox
from m2det import build_net
from data import BaseTransform
from utils.core import *
from utils.pycocotools.coco import COCO
from utils.box_utils import _to_color, draw_detection
from torchsummary import summary

parser = argparse.ArgumentParser(description='M2Det Testing')
parser.add_argument('-c', '--config', default='configs/m2det320_vgg.py', type=str)
parser.add_argument('-f', '--directory', default='imgs/', help='the path to demo images')
parser.add_argument('-m', '--trained_model', default=None, type=str, help='Trained state_dict file path to open')
parser.add_argument('--show', action='store_true', help='Whether to display the images')
args = parser.parse_args()

print_info(' ----------------------------------------------------------------------\n'
           '|                       M2Det Demo Program                             |\n'
           ' ----------------------------------------------------------------------', ['yellow','bold'])


#------------------------------------- Define --------------------------------------#
global cfg
cfg = Config.fromfile(args.config)
anchor_config = anchors(cfg)
print_info('The Anchor info: \n{}'.format(anchor_config))
priorbox = PriorBox(anchor_config)
net = build_net('test',
                size = cfg.model.input_size,
                config = cfg.model.m2det_config)
init_net(net, cfg, args.trained_model)
print_info('===> Finished constructing and loading model',['yellow','bold'])

net.eval()
with torch.no_grad():
    priors = priorbox.forward()
    if cfg.test_cfg.cuda:
        net = net.cuda()
        priors = priors.cuda()
        cudnn.benchmark = True
    else:
        net = net.cpu()

_preprocess = BaseTransform(cfg.model.input_size, cfg.model.rgb_means, (2, 0, 1))
detector = Detect(cfg.model.m2det_config.num_classes, cfg.loss.bkg_label, anchor_config)

base = int(np.ceil(pow(cfg.model.m2det_config.num_classes, 1. / 3)))
colors = [_to_color(x, base) for x in range(cfg.model.m2det_config.num_classes)]
cats = [_.strip().split(',')[-1] for _ in open('data/coco_labels.txt','r').readlines()]
labels = tuple(['__background__'] + cats)


#------------------------------------- RUN --------------------------------------#
im_path = args.directory
im_fnames = sorted((fname for fname in os.listdir(im_path) if os.path.splitext(fname)[-1] == '.jpg'))
im_fnames = (os.path.join(im_path, fname) for fname in im_fnames)


for i, fname in enumerate(im_fnames):
    if 'm2det' in fname: continue # ignore the detected images

    #read image, and pre-porcess
    image = cv2.imread(fname, cv2.IMREAD_COLOR)
    w,h = image.shape[1],image.shape[0]
    img = _preprocess(image).unsqueeze(0)
    
    if cfg.test_cfg.cuda:
        img = img.cuda()
    scale = torch.Tensor([w,h,w,h])

    #M2Det body
    out = net(img)
    #out[0].size()= torch.Size([1, 32760, 4]) --> boxes
    #out[1].size()= torch.Size([32760, 81])   --> scores
    #out[2] out of index

#    #https://www.jianshu.com/p/2cb1a4d25b4f\
#    #memory limitied, not ok for running
#    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
#    model = net.to(device)
#    summary(model, (3, 512, 512))

    #box
    boxes, scores = detector.forward(out, priors)
    boxes = (boxes[0]*scale).cpu().numpy()
    scores = scores[0].cpu().numpy()
    allboxes = []
    for j in range(1, cfg.model.m2det_config.num_classes):
        inds = np.where(scores[:,j] > cfg.test_cfg.score_threshold)[0]
        if len(inds) == 0:
            continue
        c_bboxes = boxes[inds]
        c_scores = scores[inds, j]
        c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(np.float32, copy=False)
        soft_nms = cfg.test_cfg.soft_nms
        keep = nms(c_dets, cfg.test_cfg.iou, force_cpu = soft_nms) #min_thresh, device_id=0 if cfg.test_cfg.cuda else None)
        keep = keep[:cfg.test_cfg.keep_per_class]
        c_dets = c_dets[keep, :]
        allboxes.extend([_.tolist()+[j] for _ in c_dets])

    allboxes = np.array(allboxes)
    boxes = allboxes[:,:4]
    scores = allboxes[:,4]
    cls_inds = allboxes[:,5]
    print('\n'.join(['pos:{}, ids:{}, score:{:.3f}'.format('(%.1f,%.1f,%.1f,%.1f)' % (o[0],o[1],o[2],o[3]) \
            ,labels[int(oo)],ooo) for o,oo,ooo in zip(boxes,cls_inds,scores)]))
    im2show = draw_detection(image, boxes, scores, cls_inds, colors, labels)
    # print bbox_pred.shape, iou_pred.shape, prob_pred.shape

    if im2show.shape[0] > 1100:
        im2show = cv2.resize(im2show,
                             (int(1000. * float(im2show.shape[1]) / im2show.shape[0]), 1000))
    if args.show:
        cv2.imshow('test', im2show)
        cv2.waitKey(5000)
    cv2.imwrite('{}_m2det.jpg'.format(fname.split('.')[0]), im2show)
