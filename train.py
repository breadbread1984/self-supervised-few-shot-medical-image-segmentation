#!/usr/bin/python3

from os import mkdir, listdir;
from os.path import join, exists;
import tensorflow as tf;
from models import FewShotSegmentation;
from create_datasets import parse_function_generator;

def main():

  fewshot = FewShotSegmentation();
  optimizer = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.ExponentialDecay(1e-3, decay_steps = 110000, decay_rate = 0.95));
  checkpoint = tf.train.Checkpoint(model = fewshot, optimizer = optimizer);
  train_loss = tf.keras.metrics.Mean(name = 'train loss', dtype = tf.float32);
  train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name = 'train accuracy');
  test_loss = tf.keras.metrics.Mean(name = 'test loss', dtype = tf.float32);
  test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name = 'test accuracy');
  trainset = tf.data.TFRecordDataset('trainset.tfrecord').repeat(-1).map(parse_function_generator(with_label == True, use_superpix == True)).shuffle(batch_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE);
  testset = tf.data.TFRecordDataset('testset.tfrecord').repeat(-1).map(parse_function_generator(with_label == False, use_superpix == True)).shuffle(batch_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE);
  trainset_iter = iter(trainset);
  testset_iter = iter(testset);
  # checkpoint
  if False == exists('checkpoints'): mkdir('checkpoints');
  checkpoint.restore(tf.train.latest_checkpoint('checkpoints'));
  # log
  log = tf.summary.create_file_writer('checkpoints');
  # train
  while True:
    support, supp_label, query, query_label = next(trainset_iter);
    with tf.GradientTape() as tape:
      if tf.math.reduce_any(tf.math.logical_or(tf.math.is_nan(support), tf.math.is_inf(support))) == True:
        print('detect nan in support, skip current iterations');
      if tf.math.reduce_any(tf.math.logical_or(tf.math.is_nan(supp_label), tf.math.is_inf(supp_label))) == True:
        print('detect nan in supp_label, skip current iterations');
      if tf.math.reduce_any(tf.math.logical_or(tf.math.is_nan(query), tf.math.is_inf(query))) == True:
        print('detect nan in query, skip current iterations');
      

if __name__ == "__main__":

  assert tf.executing_eagerly();
