import argparse
import glob
import json 
import os

import h5py
import nibabel as nib 
import numpy as np 
import pandas as pd

DERIV_DIR = '/gpfs2/scratch/ocaldwe1/paths/deriv_6_1'
PREP_DIR = '/gpfs2/scratch/ocaldwe1/paths/prep'

TR = 0.8
MIN_SECONDS = 300
SEED = 42 # ??? ??? #

FD_ONLY_MASK = 'desc-filtered_motion_mask.mat'
OUTLIER_MASK = 'desc-filteredwithoutoutliers_motion_mask.mat'

# Options follow DCANLab/cifit-connectivity: FD 0.2, other options off by default
parser = argparse.ArgumentParser()
parser.add_argument('state', help='rest or nback')
parser.add_argument('--label', help='output folder name')
parser.add_argument( '--fd-threshold', type=float, default=0.2)
parser.add_argument('--remove-outliers', action='store_true', 
                    help='censor with ABCC mask that also removes outlier frames')
parser.add_argument('--minutes', type=float,
                    help='randomly keep this many minutes of good frames per subject') # DCAN option
parser.add_argument('--additional-mask', 
                    help='glob for a per-subject .txt of 0 or 1 in func folder. 0 = Drop that frame.')

args = parser.parse_args()
OUTPUT_DIR = os.path.join(PREP_DIR, args.label or args.state)

def kept_frames(mask_file):
  '''
  Return a T/F vector of frames kept at our set --fd-threshold (True = kept).

  ABCC collects motion as framewise displacement (FD, in mm) and saves for all subjects
  a motion-mask file listing frames kept/lost at FD cut-offs (from 0mm to 1mm).
  This function opens this file, finds our set cut-off (0.2mm by default) and returns vector
  detailed above.
  '''
  with h5py.File(mask_file, 'r') as f: 
    for ref in f['motion_data'][()].ravel(): 
      s = f[ref]
      if round(s['FD_threshold'][()].item(), 2) == args.fd_threshold: 
        return s['frame_removal'][()].ravel() == 0

os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(os.path.join(OUTPUT_DIR, 'options.json'), 'w') as f: 
  json.dump(vars(args), f, indent=2)
print(vars(args))

ptseries_files = sorted(glob.glob(os.path.join(
  DERIV_DIR, 'sub-*', 'ses-00A', 'func', 
f'*task-{args.state}*Gordon2014*ptseries.nii')))

# Edge-mask: All edges touching L/R amygdala or hippocampus
...
