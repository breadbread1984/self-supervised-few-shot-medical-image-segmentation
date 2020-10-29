#!/usr/bin/python3

from os import listdir;
from os.path import join, splitext, basename;
import numpy as np;
from skimage.segmentation import felzenszwalb; # graph cut
from skimage.measure import label; # label connected component
import scipy.ndimage.morphology as snm; # morphology
import SimpleITK as sitk;
import cv2;

HIST_CUT_TOP = 0.5;
NEW_SPA = [1.25, 1.25, 7.70]; # unified voxel spacing
CROP_SIZE = [256, 256];

def process_CHAOST2(dataset_root):

  for sid in listdir(dataset_root):
    # process a MR
    reader = sitk.ImageSeriesReader();
    dicom_names = reader.GetGDCMSeriesFileNames(join(dataset_root, sid, "T2SPIR", "DICOM_anon"));
    reader.SetFileNames(dicom_names);
    slices = reader.Execute();
    # process corresponding labels
    labels = dict();
    for f in listdir(join(dataset_root, sid, 'T2SPIR', 'Ground')):
      if splitext(f)[1] == '.png':
        label = cv2.imread(join(dataset_root, sid, 'T2SPIR', 'Ground', f));
        if label is None:
          print("label file %s is broken" % (join(dataset_root, sid, "T2SPIR", "Ground", labelfile)));
          continue;
      labels = [t[1] for t in sorted(labels.items())];
      labels = np.stack(labels, axis = 0);
    # filtering over bright area
    array = sitk.GetArrayFromImage(slices);
    hir = float(np.percentile(array, 100.0 - HIST_CUT_TOP));
    array[array > hir] = hir;
    filtered_slices = sitk.GetImageFromArray(array);
    filtered_slices.SetSpacing(slices.GetSpacing());
    filtered_slices.SetOrigin(slices.GetOrigin());
    filtered_slices.SetDirection(slices.GetDirection());
    # resampling the MR
    resampler = sitk.ResampleImageFilter();
    resampler.SetInterpolator(sitk.sitkLinear);
    resampler.SetOutputDirection(filtered_slices.GetDirection());
    resampler.SetOutputOrigin(filtered_slices.GetOrigin());
    mov_spacing = filtered_slices.GetSpacing();
    resampler.SetOutputSpacing(NEW_SPA);
    RES_COE = np.array(mov_spacing) * 1.0 / np.array(new_spacing);
    new_size = np.array(filtered_slices.GetSize()) * RES_COE;
    resampler.SetSize([int(sz + 1) for sz in new_size]);
    resampled_slices = resampler.Execute(filtered_slices);
    # crop out rois
    array = sitk.GetArrayFromImage(resampled_slices); # array.shape = (slice number, height, width)
    array = np.transpose(array, (1,2,0)); # array.shape = (height, width, slice number)
    expand_cropsize = [x + 1 for x in CROP_SIZE] + [array.shape[-1]]; # expand_cropsize = (256 + 1, 256 + 1, slice number)
    image_patch = np.ones(expand_cropsize) * np.min(array);
    half_size = tuple(np.array(expand_cropsize) // 2); # ((256 + 1) / 2, (256 + 1) / 2, slice_number / 2)
    min_idx = [0, 0, 0];
    max_idx = array.shape;
    bias_start = [0, 0, 0];
    bias_end = [0, 0, 0];
    reference_ctr_idx = [array.shape[0] // 2, array.shape[1] // 2]; # (h / 2, w / 2)
    for i in range(2):
      bias_start[i] = np.min([half_size[i], reference_ctr_idx[i]]);
      bias_end[i] = np.min([half_size[i], array.shape[i] - reference_ctr_idx[i]]);
      min_idx[i] = reference_ctr_idx[i] - bias_start[i];
      max_idx[i] = reference_ctr_idx[i] + bias_end[i];
    image_patch[half_size[0] - bias_start[0]:half_size[0] + bias_end[0], \
                half_size[1] - bias_start[1]:half_size[1] + bias_end[1], ...] = \
    array[reference_ctr_idx[0] - bias_start[0]:reference_ctr_idx[0] + bias_end[0],
          reference_ctr_idx[1] - bias_start[1]:reference_ctr_idx[1] + bias_end[1], ...];
    image_patch = image_patch[0:CROP_SIZE[0], 0:CROP_SIZE[1],:];
    array = np.transpose(image_patch, (2, 0, 1));
    cropped_slices = sitk.GetImageFromArray(array);
    cropped_slices.SetSpacing(filtered_slices.GetSpacing());
    cropped_slices.SetOrigin(filtered_slices.GetOrigin());
    cropped_slices.SetDirection(filtered_slices.GetDirection());
    # TODO

def process_SABS(dataset_root):
    
  pass;

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

if __name__ == "__main__":
    
  from sys import argv;
  if len(argv) != 3:
    print("Usage: %s (CHAOST2|SABS) <dataset path>" % (argv[0]));
    exit(1);
  if argv[1] not in ['CHAOST2', 'SABS']:
    print('unknown dataset!');
    exit(1);
  dataset_root = argv[2];
  if argv[1] == 'CHAOST2':
    process_CHAOST2(dataset_root);
  elif argv[1] == 'SABS':
    process_SABS(dataset_root);
