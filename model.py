# -*- coding: utf-8 -*-
"""
@author: wi-ith
"""
import tensorflow as tf
import tensorflow.contrib as tc
from tensorflow.contrib.framework.python.ops import arg_scope

_WEIGHT_DECAY = 1e-4

FLAGS = tf.app.flags.FLAGS

class resnet_18(object):
    def __init__(self, is_training=True, input_size=224):
        self.input_size = input_size
        self.is_training = is_training
        self.normalizer = tc.layers.batch_norm
        self.bn_params = {'is_training': self.is_training,
                          'scale': True,
                          'center': True,
                          'decay': 0.9997,
                          'epsilon': 0.001,
                          }

    def _residual_block_first(self, x, out_channel, strides, name="unit"):
        in_channel = x.get_shape().as_list()[-1]
        with tf.variable_scope(name) as scope:
            print('\tBuilding residual unit: %s' % scope.name)

            # Shortcut connection
            if in_channel == out_channel:
                if strides == 1:
                    shortcut = tf.identity(x)
                else:
                    shortcut = tf.nn.max_pool(x, [1, strides, strides, 1], [1, strides, strides, 1], 'VALID')
            else:
                shortcut = tc.layers.conv2d(x, out_channel, 1,
                                     stride=strides,
                                     activation_fn=None,
                                     normalizer_fn=None,
                                     normalizer_params=None,
                                     scope='shortcut')
            # Residual
            x = tc.layers.conv2d(x, out_channel, 3,
                                  stride=strides,
                                  activation_fn=tf.nn.relu,
                                  normalizer_fn=self.normalizer,
                                  normalizer_params=self.bn_params,
                                  scope='conv_1')
            x = tc.layers.conv2d(x, out_channel, 3,
                                 stride=1,
                                 activation_fn=None,
                                 normalizer_fn=self.normalizer,
                                 normalizer_params=self.bn_params,
                                 scope='conv_2')
            x = x+shortcut
            x = tf.nn.relu(x)
        return x

    def _residual_block(self, x, input_q=None, output_q=None, name="unit"):
        num_channel = x.get_shape().as_list()[-1]
        with tf.variable_scope(name) as scope:
            print('\tBuilding residual unit: %s' % scope.name)
            # Shortcut connection
            shortcut = x
            # Residual
            x = tc.layers.conv2d(x, num_channel, 3,
                                  stride=1,
                                  activation_fn=tf.nn.relu,
                                  normalizer_fn=self.normalizer,
                                  normalizer_params=self.bn_params,
                                  scope='conv_1')
            x = tc.layers.conv2d(x, num_channel, 3,
                                 stride=1,
                                 activation_fn=None,
                                 normalizer_fn=self.normalizer,
                                 normalizer_params=self.bn_params,
                                 scope='conv_2')
            x = x+shortcut
            x = tf.nn.relu(x)
        return x


    def _build_model(self, image):
        self.i = 0
        filters = [64, 64, 128, 256, 512]
        kernels = [7, 3, 3, 3, 3]
        strides = [2, 0, 2, 2, 2]
        with arg_scope([tc.layers.conv2d],
                       weights_regularizer=tc.layers.l2_regularizer(_WEIGHT_DECAY)):
            with tf.variable_scope('Resnet_18', reuse=tf.AUTO_REUSE):

                with tf.variable_scope('conv1'):
                    x = tc.layers.conv2d(image, filters[0], kernels[0],
                                         stride=strides[0],
                                         activation_fn=tf.nn.relu,
                                         normalizer_fn=self.normalizer,
                                         normalizer_params=self.bn_params,
                                         scope='conv_1')
                    x = tf.nn.max_pool(x, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')

                # conv2_x
                x = self._residual_block(x, name='conv2_1')
                x = self._residual_block(x, name='conv2_2')

                # conv3_x
                x = self._residual_block_first(x, filters[2], strides[2], name='conv3_1')
                x = self._residual_block(x, name='conv3_2')

                # conv4_x
                x = self._residual_block_first(x, filters[3], strides[3], name='conv4_1')
                x = self._residual_block(x, name='conv4_2')

                # conv5_x
                x = self._residual_block_first(x, filters[4], strides[4], name='conv5_1')
                x = self._residual_block(x, name='conv5_2')

                # Logit
                with tf.variable_scope('logits') as scope:
                    x = tf.reduce_mean(x, [1, 2])
                    x = tf.reshape(x,[-1,1,1,x.shape[-1]])
                    x = tc.layers.conv2d(x, FLAGS.num_classes, 1, 1,
                                          activation_fn=None,
                                          normalizer_fn=None,
                                          normalizer_params=None)

                    x = tf.reshape(x, [-1, FLAGS.num_classes])

                logits = x
                return logits


