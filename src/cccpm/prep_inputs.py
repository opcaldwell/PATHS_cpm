'''
Second step in the prep for CPM. Will run after prep_connectomes.py. 
It matches each subject's connectome to their ABCD 6.1 tables and Brieant 
factor scores, then filters to a final sample of one child per family with 
complete data. It also creates the site-stratified 10-fold CV splits 
(100 repeats) for the final sample.
'''

import argparse
import os
import shutil

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from cccpm.splits import FixedSplits

# Options: state (rest or nback) and --label, which must match the prep_connectomes.py run
parser = argparse.ArgumentParser()
parser.add_argument('state', help='rest or nback')
parser.add_argument('--label', help='prep folder name from prep_connectomes.py (default: state)')
args = parser.parse_args()
STATE = args.state

# Input = ABCD 6.1 tables, Brieant factor scores and prep_connectomes.py output; output = driver inputs
TAB_DIR = '/gpfs2/scratch/ocaldwe1/paths/tab_6/dairc/rawdata/phenotype'
BRIEANT_FILE = '/gpfs2/scratch/ocaldwe1/paths/brieant_factors.csv'
PREP_DIR = os.path.join('/gpfs2/scratch/ocaldwe1/paths/prep', args.label or STATE)
OUTPUT_DIR = os.path.join('/gpfs2/scratch/ocaldwe1/paths/cpm_input', args.label or STATE)

# ABCD inclusion flag and mean FD (before censoring) for each state
QC_COLUMNS = {'rest': ('mr_y_qc__incl__rsfmri_indicator', 'mr_y_qc__mot__rsfmri__mot_mean'),
              'nback': ('mr_y_qc__incl__tfmri__nback_indicator', 'mr_y_qc__mot__tfmri__nback__mot_mean')}
N_FOLDS = 10
N_REPEATS = 100
SEED = 42 # Fixes which child is kept per family and how subjects are split into folds

# 11 CPM targets (Brieant F1-F10, then CBCL depression) and the covariates
TARGETS = [f'F{k}' for k in range(1, 11)] + ['cbcl_dep']
COVARIATES = ['site', 'age', 'sex', 'mean_fd']

def read_table(name, columns):
  '''
  Return the chosen columns of one ABCD table, one row per participant at baseline.

  Each ABCD 6.1 table is a .tsv file in TAB_DIR. Values are read as text and converted to numbers later.
  Tables with a session_id column (measured at several visits) are cut to the baseline visit, ses-00A;
  tables without one (fixed values, e.g. sex and family ID) already have one row per participant.
  '''
  df = pd.read_csv(os.path.join(TAB_DIR, f'{name}.tsv'), sep='\t', dtype=str,
                   usecols=lambda c: c in ['participant_id', 'session_id'] + columns)
  if 'session_id' in df:
    df = df[df['session_id'] == 'ses-00A']
  return df.set_index('participant_id')[columns]

# One table: a row per participant, a column per variable, matched on participant_id; short column names
incl_col, fd_col = QC_COLUMNS[STATE]
data = pd.concat([
  read_table('ab_g_dyn', ['ab_g_dyn__design_site', 'ab_g_dyn__visit_age']),
  read_table('ab_g_stc', ['ab_g_stc__cohort_sex', 'ab_g_stc__design_id__fam']),
  read_table('mr_y_qc__incl', [incl_col]),
  read_table('mr_y_qc__mot', [fd_col]),
  read_table('mh_p_cbcl', ['mh_p_cbcl__dsm__dep_sum']),
  pd.read_csv(BRIEANT_FILE, index_col='participant_id'),
], axis=1).rename(columns={
  'ab_g_dyn__design_site': 'site', 'ab_g_dyn__visit_age': 'age',
  'ab_g_stc__cohort_sex': 'sex', 'ab_g_stc__design_id__fam': 'family',
  incl_col: 'include', fd_col: 'mean_fd', 'mh_p_cbcl__dsm__dep_sum': 'cbcl_dep'})
# Text to numbers; anything non-numeric (e.g. 'n/a') becomes missing
for col in ['age', 'include', 'mean_fd', 'cbcl_dep']:
  data[col] = pd.to_numeric(data[col], errors='coerce')

# Sample flow, in order (numbers for the Methods flowchart)
# Start: everyone prep_connectomes.py found with a ptseries and a motion mask
qc = pd.read_csv(os.path.join(PREP_DIR, 'qc.csv'))
ids = pd.read_csv(os.path.join(PREP_DIR, 'connectome_ids.csv'))['participant_id']
sample = data.reindex(qc['participant_id'])
print('ptseries + motion mask:', len(sample))
# Keep those ABCD marks as usable for this state
sample = sample[sample['include'] == 1]
print('ABCD inclusion flag:', len(sample))
# Keep those with >= 300 s after censoring (the subjects in connectomes.npy)
sample = sample[sample.index.isin(ids)]
print('enough frames after censoring:', len(sample))
# Keep those with every target, covariate and a family ID
sample = sample.dropna(subset=TARGETS + COVARIATES + ['family'])
print('complete targets and covariates:', len(sample))
# Keep one random child per family (shuffle fixed by SEED), then put back in ID order
sample = sample.sample(frac=1, random_state=SEED).drop_duplicates('family').sort_index()
print('one child per family:', len(sample))

# Site-stratified 10-fold CV, 100 repeats: fold_ids[r, i] = test fold of subject i in repeat r
fold_ids = np.empty((N_REPEATS, len(sample)), dtype=int)
for r in range(N_REPEATS):
  # New shuffle each repeat (seed 42, 43, ...); every fold gets a similar mix of sites
  cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED + r)
  for k, (_, test) in enumerate(cv.split(sample, sample['site'])):
    fold_ids[r, test] = k

# Save the driver's inputs, all in the same subject order
os.makedirs(OUTPUT_DIR, exist_ok=True)
# Row of each final subject in connectomes.npy
rows = pd.Series(np.arange(len(ids)), index=ids)[sample.index].values
# Connectomes for the final sample only (mmap_mode reads just those rows from disk)
np.save(os.path.join(OUTPUT_DIR, 'X.npy'), np.load(os.path.join(PREP_DIR, 'connectomes.npy'), mmap_mode='r')[rows])
sample[TARGETS].to_csv(os.path.join(OUTPUT_DIR, 'targets.csv'), index=False)
sample[COVARIATES].to_csv(os.path.join(OUTPUT_DIR, 'covariates.csv'), index=False)
# Folds plus the subject IDs they refer to
FixedSplits(fold_ids, sample.index).save(os.path.join(OUTPUT_DIR, 'folds.npz'))
# Edge mask from prep_connectomes.py, unchanged
shutil.copy(os.path.join(PREP_DIR, 'edge_mask.npy'), OUTPUT_DIR)
