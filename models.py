#!/usr/bin/python3

import tensorflow as tf;

def ALPNet(height, width, channel = 2048, mode = 'gridconv+', thresh = 0.95, name = None):

  assert mode in ['mask', 'gridconv', 'gridconv+'];
  query = tf.keras.Input((height, width, channel)); # query.shape = (qn, h, w, c)
  support = tf.keras.Input((height, width, channel)); # support.shape = (nshot, h, w, c)
  labels = tf.keras.Input((height, width, 1)); # labels.shape = (nshot, h, w, 1) with foreground value 1 and background value 0
  if mode == 'mask':
    # convolute query tensor with a single foreground prototype vector
    proto = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x[0] * x[1], axis = (1, 2)) / (tf.math.reduce_sum(x[1], axis = (1, 2)) + 1e-5))([support, labels]); # proto.shape = (nshot, c)
    proto = tf.keras.layers.Lambda(lambda x: tf.reshape(tf.math.reduce_mean(x, axis = 0), (1, 1, 1, -1)))(proto); # proto.shape = (1, 1, 1, c)
    pred_mask = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x[0] * x[1], axis = -1) / tf.maximum(tf.norm(x[0], axis = -1) * tf.norm(x[1], axis = -1), 1e-4))([query, proto]); # pred_mask.shape = (qn, h, w)
    return tf.keras.Model(inputs = (query, support, labels), outputs = pred_mask, name = name);
  else:
    # get foreground prototype vectors of down sampled input tensor
    n_sup = tf.keras.layers.AveragePooling2D(pool_size = (4, 4))(support); # n_sup.shape = (nshot, nh, nw, c)
    n_sup = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[-1])))(n_sup); # n_sup.shape = (nshot * nh * nw, c)
    n_label = tf.keras.layers.AveragePooling2D(pool_size = (4, 4))(labels); # n_label.shape = (nshot, nh, nw, 1)
    n_label = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1,)))(n_label); # n_label.shape = (nshot * nh * nw)
    fg = tf.keras.layers.Lambda(lambda x, t: tf.math.greater(x, t), arguments = {'t': thresh})(n_label); # mask.shape = (nshot * nh * nw)
    protos = tf.keras.layers.Lambda(lambda x: tf.boolean_mask(x[0], x[1]))([n_sup, fg]); # protos.shape = (n, c)
    # normalize query tensor
    qry = tf.keras.layers.Lambda(lambda x: x / tf.maximum(tf.norm(x, axis = -1, keepdims = True), 1e-4))(query); # qry.shape = (qn, h, w, c)
    if mode == 'gridconv':
      # convolute query tensor with downsampled foreground prototype vectors
      protos = tf.keras.layers.Lambda(lambda x: x / tf.maximum(tf.norm(x, axis = -1, keepdims = True), 1e-4))(protos); # protos.shape = (n, c)
      filters = tf.keras.layers.Lambda(lambda x: tf.expand_dims(tf.expand_dims(tf.transpose(x, (1, 0)), axis = 0), axis = 0))(protos); # filters.shape = (1, 1, c, n)
      dists = tf.keras.layers.Lambda(lambda x: tf.nn.conv2d(input = x[0], filters = x[1], strides = (1, 1), padding = 'VALID') * 20)([qry, filters]); # dists.shape = (qn, h, w, n)
      pred_grid = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(tf.nn.softmax(x, axis = -1) * x, axis = -1))(dists); # pred_grid.shape = (qn, h, w)
      return tf.keras.Model(inputs = (query, support, labels), outputs = pred_grid, name = name);
    elif mode == 'gridconv+':
      # convolute query tensor with prototype vectors from original resolution input tensor and downsampled input tensor
      glb_proto = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x[0] * x[1], axis = (1, 2)) / tf.math.reduce_sum(x[1], axis = (1, 2)) + 1e-5)([support, labels]); # glb_proto.shape = (nshot, c)
      merge_proto = tf.keras.layers.Concatenate(axis = 0)([protos, glb_proto]); # merge_proto.shape = (n + nshot, c)
      pro_n = tf.keras.layers.Lambda(lambda x: x / tf.maximum(tf.norm(x, axis = -1, keepdims = True), 1e-4))(merge_proto); # pro_n.shape = (n + nshot, c)
      filters = tf.keras.layers.Lambda(lambda x: tf.expand_dims(tf.expand_dims(tf.transpose(x, (1, 0)), axis = 0), axis = 0))(pro_n); # filters.shape = (1, 1, c, n + nshot)
      dists = tf.keras.layers.Lambda(lambda x: tf.nn.conv2d(input = x[0], filters = x[1], strides = (1, 1), padding = 'VALID') * 20)([qry, filters]); # dists.shape = (qn, h, w, n + nshot)
      pred_grid = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(tf.nn.softmax(x, axis = -1) * x, axis = -1))(dists); # pred_grid.shape = (qn, h, w)
      return tf.keras.Model(inputs = (query, support, labels), outputs = pred_grid, name = name);
    else:
      raise Exception('unknown mode!');

class FewShotSegmentation(tf.keras.Model):

  def __init__(self, height, width, thresh = 0.95, name = 'few_shot_segmentation', **kwargs):

    super(FewShotSegmentation, self).__init__(name = name, ** kwargs);
    self.resnet101 = tf.keras.applications.ResNet101(include_top = False, weights = 'imagenet');
    self.thresh = thresh;

  def call(self, inputs):

    # query.shape = (qn, h, w, 3)
    # support.shape = (nshot, h, w, 3)
    # fg.shape = (nshot, h, w, nway)
    # bg.shape = (nshot, h, w, nway = 1)
    # with_loss.shape = ()
    query, support, fg, bg, with_loss = inputs;
    assert bg.shape[-1] == 1; # because background can only has one mask
    assert type(with_loss) is bool;
    imgs_concat = tf.keras.layers.Concatenate(axis = 0)([support, query]); # imgs_concat.shape = (nshot + qn, h, w, 3)
    img_fts = self.resnet101(imgs_concat); # img_fts.shape = (nshot + 1, nh, nw, 2048)
    supp_fts, qry_fts = tf.keras.layers.Lambda(lambda x: tf.split(x, (-1, query.shape[0]), axis = 0))(img_fts); # supp_fts.shape = (nshot, nh, nw, 2048), qry_fts.shape = (qn, nh, nw, 2048)
    ds_fg = tf.keras.layers.Lambda(lambda x: tf.image.resize(x[0], size = x[1].shape[1:3]))([fg, img_fts]); # ds_fg.shape = (nshot, nh, nw, nway)
    ds_bg = tf.keras.layers.Lambda(lambda x: tf.image.resize(x[0], size = x[1].shape[1:3]))([bg, img_fts]); # ds_bg.shape = (nshot, nh, nw, nway = 1)
    scores = list();
    bg_raw_score = ALPNet(qry_fts.shape[1], qry_fts.shape[2], qry_fts.shape[3], mode = 'gridconv', thresh = self.thresh)([qry_fts, supp_fts, ds_bg[..., 0]]); # bg_raw_score.shape = (qn, nh, nw)
    scores.append(bg_raw_score);
    for i in range(ds_fg.shape[-1]):
      maxval = tf.keras.layers.Lambda(lambda x: tf.math.reduce_max(tf.nn.avg_pool2d(tf.expand_dims(x, axis = -1), (4, 4), strides = (1, 1), padding = 'VALID')))(ds_fg[..., i]);
      fg_raw_score = ALPNet(qry_fts.shape[1], qry_fts.shape[2], qry_fts.shape[3], mode = 'gridconv+', thresh = self.thresh)([qry_fts, supp_fts, ds_fg[..., i]]) if maxval > self.thresh \
        else ALPNet(qry_fts.shape[1], qry_fts.shape[2], qry_fts.shape[3], mode = 'mask', thresh = self.thresh)([qry_fts, supp_fts, ds_fg[..., i]]); # fg_raw_score.shape = (qn, nh, nw)
      scores.append(fg_raw_score);
    scores = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis = -1))(scores); # scores.shape = (qn, nh, nw, 1 + foreground number)
    pred = tf.keras.layers.Lambda(lambda x: tf.image.resize(x[0], x[1].shape[1:3]))([scores, fg]); # us_scores.shape = (qn, h, w, 1 + foreground number)
    # get align_loss
    if with_loss:
      pred_cls = tf.keras.layers.Lambda(lambda x: tf.math.argmax(x, axis = -1))(pred); # pred_cls.shape = (qn, h, w)
      query_label = tf.one_hot(pred_cls, depth = pred.shape[-1], axis = -1); # pred_cls.shape = (qn, h, w, 1 + foreground number)
      query_bg, query_fg = tf.kersa.layers.Lambda(lambda x: tf.split(x, (1, -1), axis = -1))(query_label); # query_bg.shape = (qn, h, w, 1), query_fg.shape = (qn, h, w, foreground number)
      bg_raw_score = ALPNet(supp_fts.shape[1], supp_fts.shape[2], supp_fts.shape[3], mode = 'gridconv', thresh = self.thresh)([supp_fts, qry_fts, query_bg[..., 0]]); # bg_raw_score.shape = (nshot, nh, nw)
      for i in range(query_fg.shape[-1]):
        maxval = tf.keras.layers.Lambda(lambda x: tf.math.reduce_max(tf.nn.avg_pool2d(tf.expand_dims(x, axis = -1), (4, 4), strides = (1, 1), padding = 'VALID')))(query_fg[..., i]);
        fg_raw_score = ALPNet(supp_fts.shape[1], supp_fts.shape[2], supp_fts.shape[3], mode = 'gridconv+', thresh = self.thresh)([supp_fts, qry_fts, query_fg[..., i]]) if maxval > self.thresh \
          else ALPNet(supp_fts.shape[1], supp_fts.shape[2], supp_fts.shape[3], mode = 'mask', thresh = self.thresh)([supp_fts, qry_fts, query_fg[..., i]]); # fg_raw_score.shape = (nshot, nh, nw)
      
    return pred;

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
  fss = FewShotSegmentation(480, 640);
  q = np.random.normal(size = (8, 480, 640, 3));
  s = np.random.normal(size = (10, 480, 640, 3));
  fg = np.random.randint(low = 0, high = 2, size = (10, 480, 640, 10));
  bg = np.random.randint(low = 0, high = 2, size = (10, 480, 640, 1));
  pred = fss([q, s, fg, bg]);
  fss.save_weights('fss.h5');
