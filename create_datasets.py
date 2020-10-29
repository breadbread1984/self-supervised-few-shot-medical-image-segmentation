#!/usr/bin/python3

from os import listdir;
from os.path import join, splitext;
import numpy as np;
from skimage.segmentation import felzenszwalb; # graph cut
from skimage.measure import label; # label connected component
import scipy.ndimage.morphology as snm; # morphology
import dicom;
import simpleITK as sitk;
import cv2;

def process_CHAOST2(dataset_root):

  for sid in listdir(dataset_root):
    # 1) process data
    for f in listdir(join(dataset_root, sid, "T2SPIR", "DICOM_anon")):
      if splitext(f)[1] == '.dcm':
        slice = dicom.read_file(join(dataset_root, sid, "T2SPIR", "DICOM_anon", f));
        # TODO
    # 2) process labels
    for f in listdir(join(dataset_root, sid, "T2SPIR", "Ground")):
      if splitext(f)[1] == '.png':
        label = cv2.imread(join(dataset_root, sid, "T2SPIR", "Ground", f));
        if label is None:
          print("label file %s is broken" % (join(dataset_root, sid, "T2SPIR", "Ground", f)));
          continue;
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
