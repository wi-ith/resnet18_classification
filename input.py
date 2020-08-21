# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 01:05:20 2020

@author: wi-ith
"""

import tensorflow as tf

import numpy as np

import crop_pad

import os

FLAGS = tf.app.flags.FLAGS

# image net
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

def decode_jpeg(image_buffer, channels, scope=None):
    """Decode a JPEG string into one 3-D float image Tensor.

    Args:
      image_buffer: scalar string Tensor.
      scope: Optional scope for name_scope.
    Returns:
      3-D float Tensor with values ranging from [0, 1).
    """
    with tf.name_scope(values=[image_buffer], name=scope,
                       default_name='decode_jpeg'):

        image = tf.image.decode_jpeg(image_buffer, channels)
        return image

def random_color(image):
    # Lighting noise (AlexNet-style PCA-based noise) from torch code
    # https://github.com/facebook/fb.resnet.torch/blob/master/datasets/transforms.lua
    alphastd = 0.1
    eigval = np.array([0.2175, 0.0188, 0.0045], dtype=np.float32)
    eigvec = np.array([[-0.5675,  0.7192,  0.4009],
                       [-0.5808, -0.0045, -0.8140],
                       [-0.5836, -0.6948,  0.4203]], dtype=np.float32)

    alpha = tf.random_normal([3, 1], mean=0.0, stddev=alphastd)
    rgb = alpha * (eigval.reshape([3, 1]) * eigvec)
    image = image + tf.reduce_sum(rgb, axis=0)
    return image

def parse_tfrecords(example_serialized):
    """
    returns:
        image_buffer : decoded image file
        class_id : 1D class tensor
        bbox : 2D bbox tensor

    """
    # Dense features in Example proto.

    context, sequence = tf.parse_single_sequence_example(
        example_serialized,
        context_features={
            'image/encoded':
                tf.FixedLenFeature((), dtype=tf.string),
            'image/format':
                tf.FixedLenFeature((), dtype=tf.string),
            'image/class/label':
                tf.FixedLenFeature([], dtype=tf.int64),
            'image/height':
                tf.FixedLenFeature([], dtype=tf.int64),
            'image/width':
                tf.FixedLenFeature([], dtype=tf.int64),
        },
    )

    image_encoded = context['image/encoded']
    image_encoded = decode_jpeg(image_encoded, 3)

    img_width = context['image/width']
    img_height = context['image/height']

    class_id = context['image/class/label']

    return image_encoded, class_id, img_width, img_height


def distorted_inputs(batch_size):
    if not batch_size:
        batch_size = FLAGS.batch_size

    with tf.device('/cpu:0'):
        images_batch, labels_batch = _get_images_labels(batch_size, 'train', FLAGS.num_readers)

    return images_batch, labels_batch


def inputs(batch_size):
    if not batch_size:
        batch_size = FLAGS.batch_size

    with tf.device('/cpu:0'):
        images_batch, labels_batch = _get_images_labels(batch_size, 'validation', 1)

    return images_batch, labels_batch


def _get_images_labels(batch_size, split, num_readers, num_preprocess_threads=None):
    """Returns Dataset for given split."""
    with tf.name_scope('process_batch'):
        dataset_dir = FLAGS.tfrecords_dir
        tfrecords_list = tf.gfile.Glob(os.path.join(dataset_dir, '*' + split + '*'))

    if tfrecords_list is None:
        raise ValueError('There are not files')

    if split == 'train':
        filename_queue = tf.train.string_input_producer(tfrecords_list,
                                                        shuffle=True,
                                                        capacity=16)
    elif split == 'validation':
        filename_queue = tf.train.string_input_producer(tfrecords_list,
                                                        shuffle=False,
                                                        capacity=1)
    else:
        raise ValueError('Not appropriate split name')

    if num_preprocess_threads is None:
        num_preprocess_threads = FLAGS.num_preprocess_threads

    if num_preprocess_threads % 4:
        raise ValueError('Please make num_preprocess_threads a multiple '
                         'of 4 (%d % 4 != 0).', num_preprocess_threads)

    if num_readers is None:
        num_readers = FLAGS.num_readers

    if num_readers < 1:
        raise ValueError('Please make num_readers at least 1')

    examples_per_shard = 300
    min_queue_examples = examples_per_shard * FLAGS.input_queue_memory_factor
    if split == 'train':
        examples_queue = tf.RandomShuffleQueue(
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples,
            dtypes=[tf.string])

    elif split == 'validation':
        examples_queue = tf.FIFOQueue(
            capacity=examples_per_shard + 3 * batch_size,
            dtypes=[tf.string])

    if num_readers > 1:
        enqueue_ops = []
        for _ in range(num_readers):
            reader = tf.TFRecordReader()
            _, value = reader.read(filename_queue)
            enqueue_ops.append(examples_queue.enqueue([value]))

        tf.train.queue_runner.add_queue_runner(
            tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))
        example_serialized = examples_queue.dequeue()
    else:
        reader = tf.TFRecordReader()
        _, example_serialized = reader.read(filename_queue)

    batch_input = []
    for thread_id in range(num_preprocess_threads):
        image_encoded, class_ids, img_width, img_height= parse_tfrecords(example_serialized)

        images, labels = image_augmentation(image_encoded,
                                            class_ids,
                                            img_width,
                                            img_height,
                                            split,
                                            thread_id)
        batch_input.append([images, labels])

    images_batch, labels_batch = tf.train.batch_join(
        batch_input,
        batch_size=batch_size,
        capacity=2 * num_preprocess_threads * batch_size)

    height = FLAGS.image_size
    width = FLAGS.image_size
    depth = 3

    images_batch = tf.cast(images_batch, tf.float32)
    images_batch = tf.reshape(images_batch, shape=[batch_size, height, width, depth])

    labels_batch = tf.cast(labels_batch, tf.int32)
    labels_batch = tf.reshape(labels_batch, shape=[batch_size])

    return images_batch, labels_batch


def image_augmentation(image_encoded, class_id, width, height, split, thread_id=0):
    if split == 'train':
        images, labels = train_augmentation(image_encoded, class_id, width, height)

    elif split == 'validation':
        images, labels = eval_augmentation(image_encoded, class_id, width, height)

    return images, labels


def train_augmentation(image_encoded, labels, width, height):


    with tf.name_scope('augmented_image'):

        image = tf.to_float(image_encoded)

        with tf.name_scope('RandomHorizontalFlip'):
            random_flip_prob = FLAGS.random_flip_prob

            def _flip_image(image):
                # flip image
                image_flipped = tf.image.flip_left_right(image)
                return image_flipped

            random = tf.random_uniform(
                [],
                minval=0,
                maxval=1,
                dtype=tf.float32,
                seed=None,
                name=None
            )

            image = tf.cond(tf.greater_equal(random, random_flip_prob),
                            lambda: image,
                            lambda: _flip_image(image))

        with tf.name_scope('RandomCrop'):
            random_coef = FLAGS.crop_prob
            random = tf.random_uniform([], minval=0, maxval=1, dtype=tf.float32)
            dst_image, labels = tf.cond(tf.greater(random_coef, random),
                                        lambda: (image, labels),
                                        lambda: crop_pad.random_crop_image(
                                            image=image,
                                            labels=labels
                                        ))

        with tf.name_scope('ResizeImage'):
            new_image = tf.image.resize_images(dst_image, tf.stack([FLAGS.image_size, FLAGS.image_size]))

        with tf.name_scope('RandomColor'):
            distorted_image = new_image / 255.

            distorted_image = tf.image.random_brightness(distorted_image, max_delta=0.4)
            distorted_image = tf.image.random_contrast(distorted_image, lower=0.6, upper=1.4)
            distorted_image = tf.image.random_saturation(distorted_image, lower=0.6, upper=1.4)

            # Lighting noise
            distorted_image = random_color(distorted_image)

        distorted_image = (distorted_image - imagenet_mean)/imagenet_std

        return distorted_image, labels


def eval_augmentation(image_encoded, labels, width, height):
    with tf.name_scope('eval_image'):
        # labels.set_shape([FLAGS.batch_size])
        width = tf.cast(width,dtype=tf.float32)
        height = tf.cast(height, dtype=tf.float32)

        with tf.name_scope('ResizeImage'):
            resize_shape = tf.cond(tf.less(width, height),
                                   lambda: ((height / width) * 256.,256.),
                                   lambda: (256.,(width / height) * 256.))
            resize_shape = tf.reshape(resize_shape, [2])
            resize_shape = tf.cast(resize_shape, dtype=tf.int32)
            new_image = tf.image.resize_images(image_encoded, resize_shape)
            resize_shape = tf.cast(resize_shape, dtype=tf.float32)
            w_ct = tf.cast(resize_shape[0] / tf.constant(2.), dtype=tf.int32)
            h_ct = tf.cast(resize_shape[1] / tf.constant(2.), dtype=tf.int32)
            offset = int(FLAGS.image_size / 2.)
            ct_crop_new_image = new_image[w_ct-offset:w_ct+offset,h_ct-offset:h_ct+offset,:]
        ct_crop_new_image.set_shape([FLAGS.image_size,FLAGS.image_size,3])
        image = ct_crop_new_image / 255.
        image = (image - imagenet_mean) / imagenet_std

        return image, labels
