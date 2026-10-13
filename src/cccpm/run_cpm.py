import os 
import sys 

import numpy as np 
import pandas as pd 
import torch 

from cccpm import CPMAnalysis, UnivariateEdgeSelection, PThreshold
from cccpm.splits import FixedSplits

DATA_DIR = '/gpfs2/scratch/ocaldwe1/paths/cpm_input'
OUTPUT_DIR = '/gpfs2/scratch/ocaldwe1/paths/cpm_results'

torch.set_num_threads(int(os.environ['SLURM_CPUS_PER_TASK']))

# One model per array task: column i of targets.csv (F1-F10, then YSR)
targets = pd.read_csv(os.path.join(DATA_DIR, 'targets.csv'))
target = targets.columns[int(os.environ['SLURM_ARRAY_TASK_ID'])]

X = np.load(os.path.join(DATA_DIR, 'X.npy'))
covariates = pd.read_csv(os.path.join(DATA_DIR, 'covariates.csv'),
                         dtype={'site':str})
site = covariates['site']
covariates = pd.get_dummies(covariates, drop_first=True, dtype=float)
dtype={'site': str})
cv = FixedSplits.load(os.path.join(DATA_DIR, 'folds.npz'))

edge_selection = UnivariateEdgeSelection(
    selection_statistic='pearson',
    selection_input='residualized', 
    edge_selection=[PThreshold(threshold=[0.05],
                                         correction=[None])],
                                         effect_size='partial')

cpm = CPMAnalysis(results_directory=os.path.join(OUTPUT_DIR, target),
                  cv=cv,
                  edge_selection=edge_selection,
                  impute_missing_values=False,
                  n_permutations=int(sys.argv[1]),
                  permutation_groups=covariates['site'],
                  pack_fold_edges=True)

cpm.run(X=X, y=targets[target].values, covariates=covariates)





                         
