#!/usr/bin/python3

from os import listdir;
from os.path import join, splitext, basename;
import numpy as np;
from skimage.segmentation import felzenszwalb; # graph cut
from skimage.measure import label; # label connected component
import scipy.ndimage.morphology as snm; # morphology
import SimpleITK as sitk;
import cv2;
import tensorflow as tf;
import tensorflow_addons as tfa;

HIST_CUT_TOP = 0.5;
NEW_SPA = [1.25, 1.25, 7.70]; # unified voxel spacing
CROP_SIZE = [256, 256];

def parse_function_generator(with_label = True, use_superpix = False):

  # with_label: dataset has label
  # use_superpix: output of the parse function uses superpixel
  if with_label == False and use_superpix == False:
    raise Exception('dataset has no label information!');
  def parse_function(serialized_example):

    feature = tf.io.parse_single_example(
      serialized_example,
      features = {
        'image': tf.io.FixedLenFeature((256, 256), dtype = tf.float32),
        'label': tf.io.FixedLenFeature((256, 256), dtype = tf.float32),
        'superpix': tf.io.FixedLenFeature((256, 256), dtype = tf.float32)
      } if with_label == True else {
        'image': tf.io.FixedLenFeature((256, 256), dtype = tf.float32),
        'superpix': tf.io.FixedLenFeature((256, 256), dtype = tf.float32)        
      }
    );
    if use_superpix:
      # choose a superpixel area
      label = tf.random.uniform(maxval = tf.cast(tf.math.reduce_max(feature['superpix']) + 1, dtype = tf.int32), dtype = tf.int32, shape = ());
      label = tf.where(tf.math.equal(feature['superpix'], tf.cast(label, dtype = tf.float32)), tf.ones_like(feature['superpix'], dtype = tf.float32), tf.zeros_like(feature['superpix'], dtype = tf.float32)); # label.shape = (256, 256)
      # pseudo label has values in {'background': 0, 'superpix foreground': 1}
    else:
      label = feature['label']; # label.shape = (256, 256)
      # real label has values in {'background': 0, 'liver': 1, 'right kidney': 2, 'left kidney': 3, 'spleen': 4}
    image_with_label = tf.stack([feature['image'], label], axis = -1); # image_with_label.shape = (256, 256, 2)
    # data augmentation
    # 1) flip
    image_with_label = tf.cond(tf.math.less(tf.random.uniform(shape = ()), 0.25), lambda: image_with_label[::-1,...], lambda: image_with_label);
    image_with_label = tf.cond(tf.math.less(tf.random.uniform(shape = ()), 0.25), lambda: image_with_label[:,::-1,...], lambda: image_with_label);
    # 2) geometric augmentation
    image_with_label = tf.expand_dims(image_with_label, axis = 0);
    # rotation
    image_with_label = tfa.image.rotate(image_with_label, tf.random.uniform(low = -30, high = 30, shape = ()));
    # translate
    image_with_label = tfa.image.translate(image_with_label, tf.random.uniform(low = -5, high = 5, shape = (2,)));
    # shear
    image_with_label = tfa.image.shear_x(image_with_label, tf.random.uniform(low = -5, high = 5, shape = ()));
    image_with_label = tfa.image.shear_y(image_with_label, tf.random.uniform(low = -5, high = 5, shape = ()));
    # zoom
    scale = tf.random.uniform(low = 0.9, high = 1.2, shape = ());
    image_with_label = tf.image.resize(image_with_label, (image_with_label.shape[1] * scale, image_with_label.shape[2] * scale));
    label = tf.clip_by_value(tf.math.round(image_with_label[...,-1]), 0, 1); # label.shape = (1, 256, 256)
    # 3) intensity augmentation
    image = image_with_label[...,0]; # image.shape = (1, 256, 256)
    image = tf.image.adjust_gamma(image, gamma = tf.random.uniform(low = 0.5, high = 1.5, shape = ()));
    image = tf.squeeze(image, axis = 0); # image.shape = (256, 256)
    label = tf.squeeze(label, axis = 0); # label.shape = (256, 256)
    return image, label;
  return parse_function;

# some helper functions
def resample_by_res(mov_img_obj, new_spacing, interpolator = sitk.sitkLinear, logging = True):
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(interpolator)
    resample.SetOutputDirection(mov_img_obj.GetDirection())
    resample.SetOutputOrigin(mov_img_obj.GetOrigin())
    mov_spacing = mov_img_obj.GetSpacing()

    resample.SetOutputSpacing(new_spacing)
    RES_COE = np.array(mov_spacing) * 1.0 / np.array(new_spacing)
    new_size = np.array(mov_img_obj.GetSize()) *  RES_COE 

    resample.SetSize( [int(sz+1) for sz in new_size] )
    if logging:
        print("Spacing: {} -> {}".format(mov_spacing, new_spacing))
        print("Size {} -> {}".format( mov_img_obj.GetSize(), new_size ))

    return resample.Execute(mov_img_obj)

def resample_lb_by_res(mov_lb_obj, new_spacing, interpolator = sitk.sitkLinear, ref_img = None, logging = True):
    src_mat = sitk.GetArrayFromImage(mov_lb_obj)
    lbvs = np.unique(src_mat)
    if logging:
        print("Label values: {}".format(lbvs))
    for idx, lbv in enumerate(lbvs):
        _src_curr_mat = np.float32(src_mat == lbv) 
        _src_curr_obj = sitk.GetImageFromArray(_src_curr_mat)
        _src_curr_obj.CopyInformation(mov_lb_obj)
        _tar_curr_obj = resample_by_res( _src_curr_obj, new_spacing, interpolator, logging )
        _tar_curr_mat = np.rint(sitk.GetArrayFromImage(_tar_curr_obj)) * lbv
        if idx == 0:
            out_vol = _tar_curr_mat
        else:
            out_vol[_tar_curr_mat == lbv] = lbv
    out_obj = sitk.GetImageFromArray(out_vol)
    out_obj.SetSpacing( _tar_curr_obj.GetSpacing() )
    if ref_img != None:
        out_obj.CopyInformation(ref_img)
    return out_obj
        
def get_label_center(label):
    nnz = np.sum(label > 1e-5)
    return np.int32(np.rint(np.sum(np.nonzero(label), axis = 1) * 1.0 / nnz))

def image_crop(ori_vol, crop_size, referece_ctr_idx, padval = 0., only_2d = True):
    """ crop a 3d matrix given the index of the new volume on the original volume
        Args:
            refernce_ctr_idx: the center of the new volume on the original volume (in indices)
            only_2d: only do cropping on first two dimensions
    """
    _expand_cropsize = [x + 1 for x in crop_size] # to deal with boundary case
    if only_2d:
        assert len(crop_size) == 2, "Actual len {}".format(len(crop_size))
        assert len(referece_ctr_idx) == 2, "Actual len {}".format(len(referece_ctr_idx))
        _expand_cropsize.append(ori_vol.shape[-1])
        
    image_patch = np.ones(tuple(_expand_cropsize)) * padval

    half_size = tuple( [int(x * 1.0 / 2) for x in _expand_cropsize] )
    _min_idx = [0,0,0]
    _max_idx = list(ori_vol.shape)

    # bias of actual cropped size to the beginning and the end of this volume
    _bias_start = [0,0,0]
    _bias_end = [0,0,0]

    for dim,hsize in enumerate(half_size):
        if dim == 2 and only_2d:
            break

        _bias_start[dim] = np.min([hsize, referece_ctr_idx[dim]])
        _bias_end[dim] = np.min([hsize, ori_vol.shape[dim] - referece_ctr_idx[dim]])

        _min_idx[dim] = referece_ctr_idx[dim] - _bias_start[dim]
        _max_idx[dim] = referece_ctr_idx[dim] + _bias_end[dim]
        
    if only_2d:
        image_patch[ half_size[0] - _bias_start[0]: half_size[0] +_bias_end[0], \
                half_size[1] - _bias_start[1]: half_size[1] +_bias_end[1], ... ] = \
                ori_vol[ referece_ctr_idx[0] - _bias_start[0]: referece_ctr_idx[0] +_bias_end[0], \
                referece_ctr_idx[1] - _bias_start[1]: referece_ctr_idx[1] +_bias_end[1], ... ]

        image_patch = image_patch[ 0: crop_size[0], 0: crop_size[1], : ]
    # then goes back to original volume
    else:
        image_patch[ half_size[0] - _bias_start[0]: half_size[0] +_bias_end[0], \
                half_size[1] - _bias_start[1]: half_size[1] +_bias_end[1], \
                half_size[2] - _bias_start[2]: half_size[2] +_bias_end[2] ] = \
                ori_vol[ referece_ctr_idx[0] - _bias_start[0]: referece_ctr_idx[0] +_bias_end[0], \
                referece_ctr_idx[1] - _bias_start[1]: referece_ctr_idx[1] +_bias_end[1], \
                referece_ctr_idx[2] - _bias_start[2]: referece_ctr_idx[2] +_bias_end[2] ]

        image_patch = image_patch[ 0: crop_size[0], 0: crop_size[1], 0: crop_size[2] ]
    return image_patch

def copy_spacing_ori(src, dst):
    dst.SetSpacing(src.GetSpacing())
    dst.SetOrigin(src.GetOrigin())
    dst.SetDirection(src.GetDirection())
    return dst

def convert2foreground_segmentation(img, thresh = 1e-4):

  # img is the 3d image
  raw_seg = np.zeros_like(img);
  for i in range(img.shape[0]):
    # segment 3d image layer by layer with graph cut
    segs = felzenszwalb(img[i, ...], min_size = 400, sigma = 1);
    raw_seg[i, ...] = segs;
  fg_mask_vol = np.zeros_like(raw_seg);
  processed_seg_vol = np.zeros_like(raw_seg);
  for i in range(raw_seg.shape[0]):
    # only focus on area with intensity over threshold
    mask_map = np.float32(raw_seg[i, ...] > thresh);
    if mask_map.max() >= 0.999:
      # if there are some area whose intensity is over threshold
      # label connected components
      labels = label(mask_map);
      assert labels.max() != 0; # if there are non-zero value in mask_map, there must be connected components
      # choose the connected components with the maximum area
      largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1;
      # fill holes within the connected component
      mask_map = snm.binary_fill_holes(largestCC);
    # foreground mask
    fgm = mask_map;
    # superpixel masking
    raw_seg2d = np.int32(raw_seg[i, ...]);
    lbvs = np.unique(raw_seg2d);
    max_lb = lbvs.max();
    # set segmentation with label 0 with number of segmentations
    raw_seg2d[raw_seg2d == 0] = max_lb + 1;
    lbvs = list(lbvs);
    lbvs.append(max_lb);
    # leave only the foreground area's label unreset
    raw_seg2d = raw_seg2d * fgm;
    lb_new = 1;
    out_seg2d = np.zeros_like(raw_seg2d);
    for lbv in lbvs:
      if lbv == 0:
        # do nothing to masked area(background area)
        continue;
      else:
        # set foreground area with new segmentation label
        out_seg2d[raw_seg2d == lbv] = lb_new;
        lb_new += 1;
    fg_mask_vol[i] = fgm;
    processed_seg_vol[i] = out_seg2d;
  # fg_mask_vol: foreground mask with foreground 1, background 0
  # processed_seg_vol: foreground segmentation with label greater or equal 1, background 0
  return fg_mask_vol, processed_seg_vol;

def process_CHAOST2(dataset_root, with_label = True):

  writer = tf.io.TFRecordWriter('trainset.tfrecord' if with_label else 'testset.tfrecord');
  if writer is None:
    print('invalid output file!');
    exit(1);
  for sid in listdir(dataset_root):
    # read MR slices from dicom
    reader = sitk.ImageSeriesReader();
    series = reader.GetGDCMSeriesIDs(join(dataset_root, sid, "T2SPIR", "DICOM_anon"));
    dicom_names = reader.GetGDCMSeriesFileNames(join(dataset_root, sid, "T2SPIR", "DICOM_anon"), series[0]);
    reader.SetFileNames(dicom_names);
    slices = reader.Execute();
    if with_label == True:
      # read the corresponding labels
      labels = dict();
      for f in listdir(join(dataset_root, sid, 'T2SPIR', 'Ground')):
        if splitext(f)[1] == '.png':
          label = cv2.imread(join(dataset_root, sid, 'T2SPIR', 'Ground', f), cv2.IMREAD_GRAYSCALE);
          if label is None:
            print("label file %s is broken" % (join(dataset_root, sid, "T2SPIR", "Ground", f)));
            continue;
          idx = int(splitext(f)[0].split('-')[-1]);
          labels[idx] = label;
      labels = [t[1] for t in sorted(labels.items())];
      labels = np.stack(labels, axis = 0); # labels.shape = (slice number, height, width)
      for new_val, old_val in enumerate(sorted(np.unique(labels))):
        labels[labels == old_val] = new_val;
      labels = sitk.GetImageFromArray(labels);
      labels = copy_spacing_ori(slices, labels);
    # filtering over bright area
    array = sitk.GetArrayFromImage(slices);
    hir = float(np.percentile(array, 100.0 - HIST_CUT_TOP));
    array[array > hir] = hir;
    his_img_o = sitk.GetImageFromArray(array);
    his_img_o = copy_spacing_ori(slices, his_img_o);
    # resampling the MR
    res_img_o = resample_by_res(his_img_o, [NEW_SPA[0], NEW_SPA[1], NEW_SPA[2]],
                                interpolator = sitk.sitkLinear, logging = True);
    # resampling the label
    if with_label == True:
      res_lb_o = resample_lb_by_res(labels,  [NEW_SPA[0], NEW_SPA[1], NEW_SPA[2] ], interpolator = sitk.sitkLinear,
                                    ref_img = None, logging = True);
    # crop out rois
    res_img_a = sitk.GetArrayFromImage(res_img_o);
    crop_img_a = image_crop(res_img_a.transpose(1,2,0), [256, 256],
                            referece_ctr_idx = [res_img_a.shape[1] // 2, res_img_a.shape[2] //2],
                            padval = res_img_a.min(), only_2d = True).transpose(2,0,1); # (slice number, height, width)
    # how MR is normalized
    mean_val = tf.math.reduce_mean(crop_img_a, keepdims = True);
    std_val = tf.math.sqrt(tf.math.reduce_mean(tf.math.square(crop_img_a - mean_val), keepdims = True));
    crop_img_a = (crop_img_a - mean_val) / std_val;
    if with_label == True:
      res_lb_a = sitk.GetArrayFromImage(res_lb_o);
      crop_lb_a = image_crop(res_lb_a.transpose(1,2,0), [256, 256],
                             referece_ctr_idx = [res_lb_a.shape[1] // 2, res_lb_a.shape[2] //2],
                             padval = 0, only_2d = True).transpose(2,0,1); # (slice number, height, width)
    # generate foreground mask and foreground segmentation
    fg_mask_vol, processed_seg_vol = convert2foreground_segmentation(crop_img_a);
    if with_label == True: assert crop_img_a.shape[0] == crop_lb_a.shape[0];
    assert crop_img_a.shape[0] == processed_seg_vol.shape[0];
    for i in range(crop_img_a.shape[0]):
      img = crop_img_a[i];
      assert img.shape == (256, 256);
      seg = processed_seg_vol[i];
      assert seg.shape == (256, 256);
      if with_label == True:
        label = crop_lb_a[i];
        assert label.shape == (256, 256);
        trainsample = tf.train.Example(features = tf.train.Features(
          feature = {
            'image': tf.train.Feature(float_list = tf.train.FloatList(value = tf.reshape(img, (-1,)))),
            'label': tf.train.Feature(float_list = tf.train.FloatList(value = tf.reshape(label, (-1,)))),
            'superpix': tf.train.Feature(float_list = tf.train.FloatList(value = tf.reshape(seg, (-1,))))
          }
        ));
      else:
        trainsample = tf.train.Example(features = tf.train.Features(
          feature = {
            'image': tf.train.Feature(float_list = tf.train.FloatList(value = tf.reshape(img, (-1,)))),
            'superpix': tf.train.Feature(float_list = tf.train.FloatList(value = tf.reshape(seg, (-1,))))
          }
        ));
      writer.write(trainsample.SerializeToString());
  writer.close();

def process_SABS(dataset_root):
    
  pass;


if __name__ == "__main__":
    
  from sys import argv;
  if len(argv) != 4:
    print("Usage: %s (CHAOST2|SABS) <trainset path> <testset path>" % (argv[0]));
    exit(1);
  if argv[1] not in ['CHAOST2', 'SABS']:
    print('unknown dataset!');
    exit(1);
  trainset_root = argv[2];
  testset_root = argv[3];
  if argv[1] == 'CHAOST2':
    process_CHAOST2(join(trainset_root, 'MR'), True);
    process_CHAOST2(join(testset_root, 'MR'), False);
  elif argv[1] == 'SABS':
    process_SABS(trainset_root, True);
    process_SABS(testset_root, False);
