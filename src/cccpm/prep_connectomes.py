'''
This is the first of two prep steps before CPM runs. It turns each subject's Gordon timeseries 
file at baseline into one row of 61,776 (all edges) Fisher-z connectivity values. High motion frames
are removed using the ABCC motion mask and dropping subjects with under 5 minutes of data. It also 
functions to have the amy/HC edge mask, subject ID and a QC table of seconds kept, which will be 
used by prep_inputs.py to match connectomes to the phenotype data. 
'''

import argparse
import glob
import json 
import os

import h5py
import nibabel as nib 
import numpy as np 
import pandas as pd

# Input = 6.1 derivative files; output = one folder per run under prep/
DERIV_DIR = '/gpfs2/scratch/ocaldwe1/paths/deriv_6_1'
PREP_DIR = '/gpfs2/scratch/ocaldwe1/paths/prep'

TR = 0.8
MIN_SECONDS = 300 # DCANLab default; subjects with under 5 min of kept data are dropped
SEED = 42 # SEE '--minutes'. Seed is set at 42 (this value can be whatever). Sets a starting point for the rng.

# ABCC provides two motion masks per task; OUTLIER_MASK has outlier frames pre-removed.
FD_ONLY_MASK = 'desc-filtered_motion_mask.mat'
OUTLIER_MASK = 'desc-filteredwithoutliers_motion_mask.mat'

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
        
# Make the output folder and record options selected for run. 
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(os.path.join(OUTPUT_DIR, 'options.json'), 'w') as f: 
  json.dump(vars(args), f, indent=2)
print(vars(args))

# Finds every subjects baseline session ptseries file
  # ptseries: parcellated time series: Single signal per brain region per frame
# sorted() places subjects in a fixed order so outputs and '--minutes' are the same in every run. 
ptseries_files = sorted(glob.glob(os.path.join(
  DERIV_DIR, 'sub-*', 'ses-00A', 'func', 
f'*task-{args.state}*Gordon2014*ptseries.nii')))

# Edge-mask: All edges touching L/R amygdala or hippocampus
names = nib.load(ptseries_files[0]).header.get_axis(1).name # Reads header of first sub's file. .name assigns labels to 352 parcels
triu = np.triu_indices(len(names), k=1) # takes positions of upper triangle at 352 parcels (names sets 352x352) skipping the diagonal
nodes = [i for i, name in enumerate(names) # Goes through parcel names keeping positions of any in "amygdala" or "hippocampus"
         if 'amygdala' in name.lower() or 'hippocampus' in name.lower()] # Lower to ensure case-matching

# Counts total edges with any node in HC/amyg
node_mask = np.zeros((len(names), len(names)), dtype=bool) # 352 x 352 grid all False
node_mask[nodes, :] = True 
node_mask[:, nodes] = True 
np.save(os.path.join(OUTPUT_DIR, 'edge_mask.npy'), node_mask[triu])
print('edge mask nodes:', [names[i] for i in nodes], '| edges:', 
node_mask[triu].sum())

# RNG for --minutes, each subject given empty row. 
rng = np.random.default_rng(SEED) # Takes SEED = 42 as starting point.
mask_name = OUTLIER_MASK if args.remove_outliers else FD_ONLY_MASK
X = np.empty((len(ptseries_files), len(triu[0])), dtype=np.float32)
qc, ids = [], [] # Seconds kept for all subjects given a mask / those making it into X
for ptseries_file in ptseries_files: 
  # Pulls subjects folder and ID from file path; skips if no motion mask 
  func_dir = os.path.dirname(ptseries_file)
  participant_id = ptseries_file.split(os.sep)[-4]
  mask_files = glob.glob(os.path.join(func_dir, 
                                      f'*task-{args.state}*{mask_name}'))
  # Skip if no motion mask
  if not mask_files: 
    continue 

  # Keep one T/F per frame; True = frame passed FD cut-off based on ABCC motion mask
  keep = kept_frames(mask_files[0])
  # Optional addition: Drop frames marked as 0 in subject's extra 0/1 file; OFF by default
  if args.additional_mask: 
    # Load 0/1 file as T/F; True only if true in both files. 
    keep &= np.loadtxt(glob.glob(os.path.join(func_dir, 
                                              args.additional_mask))[0]).astype(bool)
  qc.append((participant_id, keep.sum() * TR))
  if keep.sum() * TR < MIN_SECONDS:
    continue 
# --minutes: only runs if option is set and subject has at least that much usable data. 
  if args.minutes and args.minutes * 60 <= keep.sum() * TR: 
    # Frame numbers of every usable frame (passing ABCC motion mask)
    good = np.flatnonzero(keep)
    # Reset keep so all are False
    keep = np.zeros_like(keep)
    # Randomly pick exactly minutes X 60 (sec) / 0.8 ('TR') of those frames and set them to True
    keep[rng.choice(good, round(args.minutes * 60 / TR), replace=False)] = True

  # Load timeseries (frames x 352 parcels) and keep only True frames
  ts = nib.load(ptseries_file).get_fdata(dtype=np.float32)[keep]
  # ts.T = parcels x frames
  # [triu] takes 61,776 unique pairs; arctanh = Fisher's z calculation; stored in next empty row in X. 
  X[len(ids)] = np.arctanh(np.corrcoef(ts.T)[triu])
  # Record subject ID, position in ids = row in X
  ids.append(participant_id)

# Save X, cut unused rows (from subjects failing motion mask(s))
np.save(os.path.join(OUTPUT_DIR, 'connectomes.npy'), X[:len(ids)])
# Subject id's given in same order as rows of X
pd.DataFrame({'participant_id': ids}).to_csv(os.path.join(OUTPUT_DIR,
                                                          'connectome_ids.csv'), index=False)
# Seconds of usable data for everyone WITH a motion mask, including those dropped for < 300s of data.
pd.DataFrame(qc, columns=['participant_id',
                          'seconds']).to_csv(os.path.join(OUTPUT_DIR, 'qc.csv'), index=False)
# Prints: files found | files with a motion mask | made it into X
print(f'{len(ptseries_files)} ptseries | {len(qc)} with motion mask | {len(ids)} with >= {MIN_SECONDS} s')

