#!/usr/bin/python3

import tensorflow as tf;

def ALPNet(height, width, channel = 2048, mode = 'gridconv+', thresh = 0.95, name = None):

  assert mode in ['mask', 'gridconv', 'gridconv+'];
  query = tf.keras.Input((height, width, channel)); # query.shape = (qn, h, w, c)
  support = tf.keras.Input((height, width, channel)); # support.shape = (nshot, h, w, c)
  labels = tf.keras.Input((height, width, 1)); # labels.shape = (nshot, h, w, 1) with foreground value 1 and background value 0
  if mode == 'mask':
    # convolute query tensor with a single foreground prototype vector (average of foregound area vectors)
    proto = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x[0] * x[1], axis = (1, 2)) / (tf.math.reduce_sum(x[1], axis = (1, 2)) + 1e-5))([support, labels]); # proto.shape = (nshot, c)
    proto = tf.keras.layers.Lambda(lambda x: tf.reshape(tf.math.reduce_mean(x, axis = 0), (1, 1, 1, -1)))(proto); # proto.shape = (1, 1, 1, c)
    pred_mask = tf.keras.layers.Lambda(lambda x: 20. * tf.math.reduce_sum(x[0] * x[1], axis = -1) / tf.maximum(tf.norm(x[0], axis = -1) * tf.norm(x[1], axis = -1), 1e-4))([query, proto]); # pred_mask.shape = (qn, h, w)
    return tf.keras.Model(inputs = (query, support, labels), outputs = pred_mask, name = name);
  else:
    # get multiple foreground prototype vectors of down sampled input tensor (all foreground area vectors)
    n_sup = tf.keras.layers.AveragePooling2D(pool_size = (4, 4))(support); # n_sup.shape = (nshot, nh, nw, c)
    n_sup = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[-1])))(n_sup); # n_sup.shape = (nshot * nh * nw, c)
    n_label = tf.keras.layers.AveragePooling2D(pool_size = (4, 4))(labels); # n_label.shape = (nshot, nh, nw, 1)
    n_label = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1,)))(n_label); # n_label.shape = (nshot * nh * nw)
    fg = tf.keras.layers.Lambda(lambda x, t: tf.math.greater(x, t), arguments = {'t': thresh})(n_label); # mask.shape = (nshot * nh * nw)
    protos = tf.keras.layers.Lambda(lambda x: tf.boolean_mask(x[0], x[1]))([n_sup, fg]); # protos.shape = (n, c)
    # normalize query tensor
    qry = tf.keras.layers.Lambda(lambda x: x / tf.maximum(tf.norm(x, axis = -1, keepdims = True), 1e-4))(query); # qry.shape = (qn, h, w, c)
    if mode == 'gridconv':
      # convolute query tensor with multiple downsampled foreground prototype vectors
      protos = tf.keras.layers.Lambda(lambda x: x / tf.maximum(tf.norm(x, axis = -1, keepdims = True), 1e-4))(protos); # protos.shape = (n, c)
      filters = tf.keras.layers.Lambda(lambda x: tf.expand_dims(tf.expand_dims(tf.transpose(x, (1, 0)), axis = 0), axis = 0))(protos); # filters.shape = (1, 1, c, n)
      dists = tf.keras.layers.Lambda(lambda x: tf.nn.conv2d(input = x[0], filters = x[1], strides = (1, 1), padding = 'VALID') * 20)([qry, filters]); # dists.shape = (qn, h, w, n)
      pred_grid = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(tf.nn.softmax(x, axis = -1) * x, axis = -1))(dists); # pred_grid.shape = (qn, h, w)
      return tf.keras.Model(inputs = (query, support, labels), outputs = pred_grid, name = name);
    elif mode == 'gridconv+':
      # convolute query tensor with multiple prototype vectors from original resolution input tensor and downsampled input tensor
      glb_proto = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x[0] * x[1], axis = (1, 2)) / tf.math.reduce_sum(x[1], axis = (1, 2)) + 1e-5)([support, labels]); # glb_proto.shape = (nshot, c)
      merge_proto = tf.keras.layers.Concatenate(axis = 0)([protos, glb_proto]); # merge_proto.shape = (n + nshot, c)
      pro_n = tf.keras.layers.Lambda(lambda x: x / tf.maximum(tf.norm(x, axis = -1, keepdims = True), 1e-4))(merge_proto); # pro_n.shape = (n + nshot, c)
      filters = tf.keras.layers.Lambda(lambda x: tf.expand_dims(tf.expand_dims(tf.transpose(x, (1, 0)), axis = 0), axis = 0))(pro_n); # filters.shape = (1, 1, c, n + nshot)
      dists = tf.keras.layers.Lambda(lambda x: tf.nn.conv2d(input = x[0], filters = x[1], strides = (1, 1), padding = 'VALID') * 20)([qry, filters]); # dists.shape = (qn, h, w, n + nshot)
      pred_grid = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(tf.nn.softmax(x, axis = -1) * x, axis = -1))(dists); # pred_grid.shape = (qn, h, w)
      return tf.keras.Model(inputs = (query, support, labels), outputs = pred_grid, name = name);
    else:
      raise Exception('unknown mode!');

def Bottleneck(input_shape, filters, stride = 1, dilation = 1):

  # NOTE: either stride or dilation can be over 1
  inputs = tf.keras.Input(input_shape);
  residual = inputs;
  results = tf.keras.layers.Conv2D(filters, (1, 1), padding = 'same', use_bias = False)(inputs);
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.ReLU()(results);
  results = tf.keras.layers.Conv2D(filters, (3, 3), padding = 'same', strides = (stride, stride), dilation_rate = (dilation, dilation), use_bias = False)(results);
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.ReLU()(results);
  results = tf.keras.layers.Conv2D(filters * 4, (1, 1), padding = 'same', use_bias = False)(results);
  results = tf.keras.layers.BatchNormalization()(results);
  if stride != 1 or inputs.shape[-1] != results.shape[-1]:
    residual = tf.keras.layers.Conv2D(filters * 4, (1, 1), padding = 'same', strides = (stride, stride), use_bias = False)(residual);
    residual = tf.keras.layers.BatchNormalization()(residual);
  results = tf.keras.layers.Add()([results, residual]);
  results = tf.keras.layers.ReLU()(results);
  return tf.keras.Model(inputs = inputs, outputs = results);

def ResNetAtrous(layer_nums = [3, 4, 6, 3], dilations = [1, 2, 1]):

  strides = [2, 2, 1];
  assert layer_nums[-1] == len(dilations);
  assert len(layer_nums) == 1 + len(strides);
  inputs = tf.keras.Input((None, None, 3));
  results = tf.keras.layers.Conv2D(64, (7, 7), strides = (2,2), padding = 'same', use_bias = False)(inputs);
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.ReLU()(results);
  results = tf.keras.layers.MaxPool2D(pool_size = (3,3), strides = (2,2), padding = 'same')(results);
  def make_layer(inputs, filters, layer_num, stride = 1, dilations = None):
    assert type(dilations) is list or dilations is None;
    results = inputs;
    for i in range(layer_num):
      results = Bottleneck(results.shape[1:], filters, stride = stride if i == 0 else 1, dilation = dilations[i] if dilations is not None else 1)(results);
    return results;
  outputs1 = make_layer(results, 64, layer_nums[0]);
  results = make_layer(outputs1, 128, layer_nums[1], stride = strides[0]);
  results = make_layer(results, 256, layer_nums[2], stride = strides[1], dilations = [1] * layer_nums[2]);
  outputs2 = make_layer(results, 512, layer_nums[3], stride = strides[2], dilations = dilations);
  return tf.keras.Model(inputs = inputs, outputs = (outputs1, outputs2));

def ResNet50Atrous():

  # NOTE: (3 + 4 + 6 + 3) * 3 + 2 = 50
  inputs = tf.keras.Input((None, None, 3));
  results = ResNetAtrous([3, 4, 6, 3], [1, 2, 1])(inputs);
  return tf.keras.Model(inputs = inputs, outputs = results, name = 'resnet50');

def ResNet101Atrous():

  # NOTE: (3 + 4 + 23 + 3) * 3 + 2 = 101
  inputs = tf.keras.Input((None, None, 3));
  results = ResNetAtrous([3, 4, 23, 3], [2, 2, 2])(inputs);
  return tf.keras.Model(inputs = inputs, outputs = results, name = 'resnet101');

def FewShotSegmentation(fg_class_num = 1, thresh = 0.95, name = 'few_shot_segmentation', pretrain = None):

  query = tf.keras.Input((None, None, 3)); # query.shape = (qn, h, w, 3)
  support = tf.keras.Input((None, None, 3)); # support.shape = (nshot, h, w, 3)
  labels = tf.keras.Input((None, None, 1 + fg_class_num)); # labels.shape = (nshot, h, w, 1 + foreground number)
  imgs_concat = tf.keras.layers.Concatenate(axis = 0)([support, query]); # imgs_concat.shape = (nshot + qn, h, w, 3)
  resnet50 = ResNet50Atrous();
  if pretrain is not None: resnet50.load_weights('pretrain.h5');
  img_fts = resnet50(imgs_concat)[1]; # img_fts.shape = (nshot + qn, nh, nw, 2048)
  img_fts = tf.keras.layers.Conv2D(filters = 256, kernel_size = (1,1))(img_fts); # img_fts.shape = (nshot + qn, nh, nw, 256)
  supp_fts, qry_fts = tf.keras.layers.Lambda(lambda x: tf.split(x[0], (tf.shape(x[1])[0], tf.shape(x[2])[0]), axis = 0))([img_fts, support, query]); # supp_fts.shape = (nshot, nh, nw, 256), qry_fts.shape = (qn, nh, nw, 256)
  ds_labels = tf.keras.layers.Lambda(lambda x: tf.image.resize(x[0], size = tf.shape(x[1])[1:3], method = tf.image.ResizeMethod.NEAREST_NEIGHBOR))([labels, img_fts]); # ds_labels.shape = (nshot, nh, nw, 1 + foreground number)
  ds_bg, ds_fg = tf.keras.layers.Lambda(lambda x: tf.split(x, (1, -1), axis = -1))(ds_labels); # ds_bg.shape = (nshot, nh, nw, 1), ds_fg.shape = (nshot, nh, nw, foreground number)
  gridconv = ALPNet(qry_fts.shape[1], qry_fts.shape[2], qry_fts.shape[3], mode = 'gridconv', thresh = thresh);
  gridconv_plus = ALPNet(qry_fts.shape[1], qry_fts.shape[2], qry_fts.shape[3], mode = 'gridconv+', thresh = thresh);
  mask = ALPNet(qry_fts.shape[1], qry_fts.shape[2], qry_fts.shape[3], mode = 'mask', thresh = thresh);
  scores = list();
  bg_raw_score = gridconv([qry_fts, supp_fts, ds_bg]); # bg_raw_score.shape = (qn, nh, nw)
  for i in range(fg_class_num):
    maxval = tf.keras.layers.Lambda(lambda x, i: tf.math.reduce_max(tf.nn.avg_pool2d(x[..., i:i+1], (4,4), strides = (1,1), padding = 'VALID')), arguments = {'i': i})(ds_fg);
    fg_raw_score = tf.keras.layers.Lambda(lambda x, i, t : tf.cond(tf.math.greater(x[3], t), lambda: gridconv_plus([x[0], x[1], x[2][...,i:i+1]]), lambda: mask([x[0], x[1], x[2][...,i:i+1]])), arguments = {'i': i, 't': thresh})([qry_fts, supp_fts, ds_fg, maxval]);
    scores.append(fg_raw_score);
  scores = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis = -1))(scores); # scores.shape = (qn, nh, nw, 1 + foreground number)
  pred = tf.keras.layers.Lambda(lambda x: tf.image.resize(x[0], tf.shape(x[1])[1:3], method = tf.image.ResizeMethod.NEAREST_NEIGHBOR))([scores, labels]); # pred.shape = (qn, h, w, 1 + foreground number)
  return tf.keras.Model(inputs = (query, support, labels), outputs = (pred, supp_fts, qry_fts));

def Loss(fg_class_num, thresh = 0.95):

  labels = tf.keras.Input((None, None, 1 + fg_class_num)); # labels.shape = (nshot, h, w, 1 + foreground number)
  pred = tf.keras.Input((None, None, 1 + fg_class_num)); # pred,.shape = (qn, h, w, 1 + foreground number)
  supp_fts = tf.keras.Input((None, None, 256)); # supp_fts.shape = (nshot, nh, nw, 256)
  qry_fts = tf.keras.Input((None, None, 256)); # qry_fts.shape = (qn, nh, nw, 256)
  pred_cls = tf.keras.layers.Lambda(lambda x: tf.math.argmax(x, axis = -1))(pred); # pred_cls.shape = (qn, h, w)
  query_label = tf.keras.layers.Lambda(lambda x, c: tf.one_hot(x, depth = c, axis = -1), arguments = {'c': 1 + fg_class_num})(pred_cls); # query_label.shape = (qn, h, w, 1 + foreground number)
  ds_query_label = tf.keras.layers.Lambda(lambda x: tf.image.resize(x[0], size = tf.shape(x[1])[1:3], method = tf.image.ResizeMethod.NEAREST_NEIGHBOR))([query_label, supp_fts]); # ds_query_label.shape = (qn, nh, nw, 1 + foreground number)
  query_bg, query_fg = tf.keras.layers.Lambda(lambda x, c: tf.split(x, (1, c), axis = -1), arguments = {'c': fg_class_num})(ds_query_label); # query_bg.shape = (qn, h, w, 1), query_fg.shape = (qn, h, w, foreground number)
  gridconv = ALPNet(supp_fts.shape[1], supp_fts.shape[2], supp_fts.shape[3], mode = 'gridconv', thresh = thresh);
  gridconv_plus = ALPNet(supp_fts.shape[1], supp_fts.shape[2], supp_fts.shape[3], mode = 'gridconv+', thresh = thresh);
  mask = ALPNet(supp_fts.shape[1], supp_fts.shape[2], supp_fts.shape[3], mode = 'mask', thresh = thresh);  
  scores = list();
  bg_raw_score = gridconv([supp_fts, qry_fts, query_bg]); # bg_raw_score.shape = (nshot, nh, nw)
  scores.append(bg_raw_score);
  for i in range(fg_class_num):
    maxval = tf.keras.layers.Lambda(lambda x, i: tf.math.reduce_max(tf.nn.avg_pool2d(x[..., i:i+1], (4,4), strides = (1,1), padding = 'VALID')), arguments = {'i': i})(query_fg);
    fg_raw_score = tf.keras.layers.Lambda(lambda x, i, t: tf.cond(tf.math.greater(x[3], t), lambda: gridconv_plus([x[0], x[1], x[2][...,i:i+1]]), lambda: mask([x[0], x[1], x[2][...,i:i+1]])), arguments = {'i': i, 't': thresh})([supp_fts, qry_fts, query_fg, maxval]);
    scores.append(fg_raw_score);
  scores = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis = -1))(scores); # scores.shape = (nshot, nh, nw, 1 + foreground number)
  supp_pred = tf.keras.layers.Lambda(lambda x: tf.image.resize(x[0], tf.shape(x[1])[1:3], method = tf.image.ResizeMethod.NEAREST_NEIGHBOR))([scores, labels]); # supp_pred.shape = (nshot, h, w, 1 + foreground)
  loss = tf.keras.losses.CategoricalCrossentropy()(labels, supp_pred);
  return tf.keras.Model(inputs = (labels, pred, supp_fts, qry_fts), outputs = loss);

if __name__ == "__main__":

  assert tf.executing_eagerly();
  import numpy as np;
  q = np.random.normal(size = (8, 480, 640, 5));
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
  fss = FewShotSegmentation(10);
  q = np.random.normal(size = (8, 480, 640, 3));
  s = np.random.normal(size = (10, 480, 640, 3));
  l = np.random.randint(low = 0, high = 2, size = (10, 480, 640, 11));
  pred = fss([q, s, l]);
  fss.save_weights('fss.h5');
