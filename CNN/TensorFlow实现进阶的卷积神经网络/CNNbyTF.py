# encoding=utf-8
# TF实现进阶的CNN
# 对weiights进行L2正则化
# 对样本进行翻转、随机剪切进行数据增强
# 在最大池化层后使用LRN层
import tensorflow as tf
import numpy
import time

max_steps = 3000
batch_size = 128


def variable_with_weight_loss(shape, stddev, w1):
    var = tf.Variable(tf.truncated_normal(shape=shape, stddev=stddev))
    if w1 is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), w1, name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return var


image_holder = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
label_holder = tf.placeholder(tf.float32, [batch_size])
