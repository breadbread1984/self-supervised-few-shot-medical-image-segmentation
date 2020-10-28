#!/usr/bin/python3

import numpy as np;
from skimage.segmentation import felzenszwalb; # graph cut
from skimage.measure import label; # label connected component
import scipy.ndimage.morphology as snm; # morphology

def gen_superpixel(img, thresh = 1e-4):

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
