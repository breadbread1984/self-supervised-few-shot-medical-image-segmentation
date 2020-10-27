#!/usr/bin/python3

import tensorflow as tf;

def ALPNet(height, width, channel = 3, mode = 'gridconv+', thresh = 0.95):

  assert mode in ['mask', 'gridconv', 'gridconv+'];
  query = tf.keras.Input((height, width, channel), batch_size = 1); # query.shape = (batch = 1, h, w, c)
  support = tf.keras.Input((height, width, channel)); # support.shape = (nshot, h, w, c)
  labels = tf.keras.Input((height, width, 1)); # labels.shape = (nshot, h, w, 1) with foreground value 1 and background value 0
  if mode == 'mask':
    # convolute query tensor with a single foreground prototype vector
    proto = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x[0] * x[1], axis = (1, 2)) / (tf.math.reduce_sum(x[1], axis = (1, 2)) + 1e-5))([support, labels]); # proto.shape = (nshot, c)
    proto = tf.keras.layers.Lambda(lambda x: tf.reshape(tf.math.reduce_mean(x, axis = 0), (1, 1, 1, -1)))(proto); # proto.shape = (1, 1, 1, c)
    pred_mask = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x[0] * x[1], axis = -1) / tf.maximum(tf.norm(x[0], axis = -1) * tf.norm(x[1], axis = -1), 1e-4))([query, proto]); # pred_mask.shape = (1, h, w)
    return tf.keras.Model(inputs = (query, support, labels), outputs = pred_mask);
  else:
    # get foreground prototype vectors of down sampled input tensor
    n_sup = tf.keras.layers.AveragePooling2D(pool_size = (4, 4))(support); # n_sup.shape = (nshot, nh, nw, c)
    n_sup = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[-1])))(n_sup); # n_sup.shape = (nshot * nh * nw, c)
    n_label = tf.keras.layers.AveragePooling2D(pool_size = (4, 4))(labels); # n_label.shape = (nshot, nh, nw, 1)
    n_label = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1,)))(n_label); # n_label.shape = (nshot * nh * nw)
    fg = tf.keras.layers.Lambda(lambda x, t: tf.math.greater(x, t), arguments = {'t': thresh})(n_label); # mask.shape = (nshot * nh * nw)
    protos = tf.keras.layers.Lambda(lambda x: tf.boolean_mask(x[0], x[1]))([n_sup, fg]); # protos.shape = (n, c)
    # normalize query tensor
    qry = tf.keras.layers.Lambda(lambda x: x / tf.maximum(tf.norm(x, axis = -1, keepdims = True), 1e-4))(query); # qry.shape = (1, h, w, c)
    if mode == 'gridconv':
      # convolute query tensor with downsampled foreground prototype vectors
      protos = tf.keras.layers.Lambda(lambda x: x / tf.maximum(tf.norm(x, axis = -1, keepdims = True), 1e-4))(protos); # protos.shape = (n, c)
      filters = tf.keras.layers.Lambda(lambda x: tf.expand_dims(tf.expand_dims(tf.transpose(x, (1, 0)), axis = 0), axis = 0))(protos); # filters.shape = (1, 1, c, n)
      dists = tf.keras.layers.Lambda(lambda x: tf.nn.conv2d(input = x[0], filters = x[1], strides = (1, 1), padding = 'VALID') * 20)([qry, filters]); # dists.shape = (1, h, w, n)
      pred_grid = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(tf.nn.softmax(x, axis = -1) * x, axis = -1))(dists); # pred_grid.shape = (1, h, w)
      return tf.keras.Model(inputs = (query, support, labels), outputs = pred_grid);
    elif mode == 'gridconv+':
      # convolute query tensor with prototype vectors from original resolution input tensor and downsampled input tensor
      glb_proto = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x[0] * x[1], axis = (1, 2)) / tf.math.reduce_sum(x[1], axis = (1, 2)) + 1e-5)([support, labels]); # glb_proto.shape = (nshot, c)
      merge_proto = tf.keras.layers.Concatenate(axis = 0)([protos, glb_proto]); # merge_proto.shape = (n + nshot, c)
      pro_n = tf.keras.layers.Lambda(lambda x: x / tf.maximum(tf.norm(x, axis = -1, keepdims = True), 1e-4))(merge_proto); # pro_n.shape = (n + nshot, c)
      filters = tf.keras.layers.Lambda(lambda x: tf.expand_dims(tf.expand_dims(tf.transpose(x, (1, 0)), axis = 0), axis = 0))(pro_n); # filters.shape = (1, 1, c, n + nshot)
      dists = tf.keras.layers.Lambda(lambda x: tf.nn.conv2d(input = x[0], filters = x[1], strides = (1, 1), padding = 'VALID') * 20)([qry, filters]); # dists.shape = (1, h, w, n + nshot)
      pred_grid = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(tf.nn.softmax(x, axis = -1) * x, axis = -1))(dists); # pred_grid.shape = (1, h, w)
      return tf.keras.Model(inputs = (query, support, labels), outputs = pred_grid);
    else:
      raise Exception('unknown mode!');

def FewShotSegmentation(height, width, thresh = 0.95):

  support = tf.keras.Input((height, width, 3)); # support.shape = (nshot, h, w, 3)
  fg = tf.keras.Input((height, width)); # fg.shape = (nshot, h, w)
  bg = tf.keras.Input((height, width)); # bg.shape = (nshot, h, w)
  query = tf.keras.Input((height, width, 3), batch_size = 1); # mask.shape = (batch = 1, h, w, 3)
  imgs_concat = tf.keras.layers.Concatenate(axis = 0)([support, query]); # imgs_concat.shape = (nshot + 1, h, w, 3)
  resnet101 = tf.keras.applications.ResNet101(include_top = False, weights = 'imagenet', input_tensor = imgs_concat);
  img_fts = resnet101.output; # img_fts.shape = (nshot + 1, nh, nw, 2048)
  supp_fts, qry_fts = tf.keras.layers.Lambda(lambda x: tf.split(x, (-1, 1), axis = 0))(img_fts); # supp_fits.shape = (nshot, nh, nw, 2048), qry_fts.shape = (1, nh, nw, 2048)
  ds_fg = tf.keras.layers.Lambda(lambda x: tf.squeeze(tf.image.resize(tf.expand_dims(x[0], axis = -1), size = x[1].shape[1:3]), axis = -1))([fg, img_fts]); # ds_fg.shape = (nshot, nh, nw)
  ds_bg = tf.keras.layers.Lambda(lambda x: tf.squeeze(tf.image.resize(tf.expand_dims(x[0], axis = -1), size = x[1].shape[1:3]), axis = -1))([bg, img_fts]); # ds_bg.shape = (nshot, nh, nw)
  bg_raw_score = ALPNet(qry_fts.shape[1], qry_fts.shape[2], qry_fts.shape[3], mode = 'gridconv', thresh = thresh)([qry_fts, supp_fts, ds_bg]);
  maxval = tf.keras.layers.Lambda(lambda x: tf.math.reduce_max(tf.nn.avg_pool2d(x, (4, 4))))(ds_fg);
  fg_raw_score = tf.keras.layers.Lambda(lambda x, t: tf.cond(tf.greater(x[0], t), 
                                                             true_fn = lambda: ALPNet(x[1].shape[1], x[1].shape[2], x[1].shape[3], mode = 'gridconv+', thresh = t)([x[1], x[2], x[3]]), 
                                                             false_fn = lambda: ALPNet(x[1].shape[1], x[1].shape[2], x[1].shape[3], mode = 'mask', thresh = t)([x[1], x[2], x[3]])), 
                                        arguments = {'t': thresh})([maxval, qry_fts, supp_fts, ds_fg]);
  

if __name__ == "__main__":

  assert tf.executing_eagerly();
  import numpy as np;
  q = np.random.normal(size = (1, 480, 640, 5));
  s = np.random.normal(size = (10, 480, 640, 5));
  l = np.random.normal(size = (10, 480, 640, 1));
  alpnet = ALPNet(480, 640, mode = 'mask');
  b = alpnet([q,s,l]);
  alpnet.save('mask.h5');
  alpnet = ALPNet(480, 640, mode = 'gridconv');
  b = alpnet([q,s,l]);
  alpnet.save('gridconv.h5');
  alpnet = ALPNet(480, 640, mode = 'gridconv+');
  b = alpnet([q,s,l]);
  alpnet.save('gridconv+.h5');
