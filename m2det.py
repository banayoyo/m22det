import numpy as np
import tensorflow as tf
from utils.layer import *

class M2Det:
    def __init__(self, inputs, is_training, num_classes):
        self.num_classes = num_classes + 1 # for background class
        self.levels = 8  #8 TUMs
        self.scales = 6  #6 scales
        self.num_priors = 9
        self.build(inputs, is_training)

    def build(self, inputs, is_training):
        with tf.variable_scope('VGG16'):
            net = inputs
            net = vgg_layer(net, is_training, 64, 2)
            net = vgg_layer(net, is_training, 128, 2)
            net = vgg_layer(net, is_training, 256, 3)
            net = vgg_layer(net, is_training, 512, 3, pooling=False)
            feature1 = net
            net = vgg_layer(net, is_training, 1024, 3)
            feature2 = net

        with tf.variable_scope('M2Det'):
            #base feature is output of FFMv1
            with tf.variable_scope('FFMv1'):
                feature1 = conv2d_layer(feature1, filters=256, kernel_size=3, strides=1)
                feature1 = tf.nn.relu(batch_norm(feature1, is_training))
                feature2 = conv2d_layer(feature2, filters=512, kernel_size=1, strides=1)
                feature2 = tf.nn.relu(batch_norm(feature2, is_training))
                #upsample
                feature2 = tf.image.resize_images(feature2, tf.shape(feature1)[1:3], 
                                                  method=tf.image.ResizeMethod.BILINEAR)
                base_feature = tf.concat([feature1, feature2], axis=3)

            #FFMv_2 and TUM, 0~7
            outs = []
            for i in range(self.levels):
                #FFMv_2
                if i == 0:
                    # 1st TUM has no concat, then should mapping from 768 to 256
                    net = conv2d_layer(base_feature, filters=256, kernel_size=1, strides=1)
                    net = tf.nn.relu(batch_norm(net, is_training))
                else:
                    #others TUM has concat, then mapping from 768 to 128
                    with tf.variable_scope('FFMv2_{}'.format(i+1)):
                        net = conv2d_layer(base_feature, filters=128, kernel_size=1, strides=1)
                        net = tf.nn.relu(batch_norm(net, is_training))
                        net = tf.concat([net, out[-1]], axis=3)

                #TUM
                with tf.variable_scope('TUM{}'.format(i+1)):
                    out = tum(net, is_training, self.scales)
                #out is 1,3,5,10,20,40, but shape is????
                outs.append(out)

            features = []
            with tf.variable_scope('SFAM'):
                for i in range(self.scales):
                    feature = tf.concat([outs[j][i] for j in range(self.levels)], axis=3)
                    #feature [1,3,5,10,20,40] [8TUMS]
                    #--squeeze
                    attention = tf.reduce_mean(feature, axis=[1, 2], keepdims=True)
                    #--excitation, 2 full-connection
                    #add a new fuul-connection layer, outputs = activation(inputs * kernel + bias)
                    #in paper, fc_1 is relu, fc_2 is sigmoid
                    #demension: 1024/64=r=16
                    attention = tf.layers.dense(inputs=feature, units=64, 
                                                activation=tf.nn.sigmoid, name='fc1_{}'.format(i+1))
                    attention = tf.layers.dense(inputs=attention, units=1024,
                                                activation=tf.nn.relu, name='fc2_{}'.format(i+1))

                    #--scale
                    feature = feature * attention
                    features.insert(0, feature)

            all_cls = []
            all_reg = []
            with tf.variable_scope('prediction'):
                for i, feature in enumerate(features):
                    print('predict[',i+1,'] shape=', feature.shape)
                    cls = conv2d_layer(feature, self.num_priors * self.num_classes, 3, 1, use_bias=True)
                    cls = batch_norm(cls, is_training) # activation function is identity
                    cls = flatten_layer(cls)
                    
                    all_cls.append(cls)
                    
                    reg = conv2d_layer(feature, self.num_priors * 4, 3, 1, use_bias=True)
                    reg = batch_norm(reg, is_training) # activation function is identity
                    reg = flatten_layer(reg)
                    all_reg.append(reg)

                    
                all_cls = tf.concat(all_cls, axis=1)
                all_reg = tf.concat(all_reg, axis=1)
                num_boxes = int(all_reg.shape[-1].value / 4)
                all_cls = tf.reshape(all_cls, [-1, num_boxes, self.num_classes])
                all_cls = tf.nn.softmax(all_cls)
                all_reg = tf.reshape(all_reg, [-1, num_boxes, 4])
                self.prediction = tf.concat([all_reg, all_cls], axis=-1)


if __name__ == '__main__':
    inputs = tf.placeholder(tf.float32, [None, 320, 320, 3])
    is_training = tf.constant(False)
    num_classes = 80
    m2det = M2Det(inputs, is_training, num_classes)
