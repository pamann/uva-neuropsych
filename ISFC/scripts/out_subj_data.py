# ARGUMENT HANDLING -------------------------------------------------------------------
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--rivanna", action="store_true", help="Use flag if running on UVAs HPC Rivanna")
parser.add_argument("--headers", action="store_true", help="Use flag for CSV headers")
parser.add_argument("--trim", action="store_true", help="Use flag for trimming noise (18TRs)")
args = parser.parse_args()
RIVANNA = args.rivanna
HEADERS = args.headers
TRIMMED = args.trim

# MAIN IMPORTS ------------------------------------------------------------------------

import warnings
import sys 
if not sys.warnoptions:
  warnings.simplefilter("ignore")
import os 
import io
import glob
import time
import copy
import csv
import numpy as np
import pandas as pd 
from datetime import datetime
from nilearn import datasets, surface, plotting, image
from nilearn.input_data import NiftiMasker, NiftiLabelsMasker
import nibabel as nib
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from scipy.stats import norm, pearsonr, zscore
from scipy.spatial.distance import squareform, pdist
from statsmodels.stats.multitest import multipletests
from nilearn.plotting import plot_glass_brain, plot_roi
from nilearn.image import resample_to_img
from helpers import (isc, isfc, bootstrap_isc, permutation_isc,
                            timeshift_isc, phaseshift_isc,
                            compute_summary_statistic, load_images,
                            load_boolean_mask, mask_images,
                            MaskedMultiSubjectData, get_func_file_names, plot_isc_data, find_ptp_ids)

# SYSTEM SETUP -------------------------------
if(RIVANNA):
  sys.path.append( '/project/morrislab/ISC/isc-tutorial-master')
  data_dir = '/project/morrislab/func_data/' 
  mask_dir = '/project/morrislab/masks' 
  dir_out = './raw_subj_data/'
  atlas_dir = "./atlas"
  atlas = atlas_dir + "/shen_1mm_268_parcellation.nii.gz"
  upper_limit_n_subjs = 100
  all_task_names = ['Nemo_C', 'Nemo_S', 'Inscapes']
  all_task_folder_names = ['Nemo_C/GM', 'Nemo_S/GM', 'Inscapes/GM']
  mask_file = mask_dir + "/MNI152_mask.nii.gz"
  all_task_data_file_names = ['sub-*_GM_NemoC_preproc.nii.gz', 'sub-*_GM_NemoS_preproc.nii.gz', 'sub-*_GM_Inscapes_preproc.nii.gz'] # Rivanna or local
  csv_file = mask_dir + "/shen_268_parcellation_networklabels.csv"
else: # assumed to be running on Pam's Windows 10 machine
  sys.path.insert(1, 'Users/Pam/Documents/isc-tutorial-master/isc-tutorial-master/isc-tutorial') # locally stored on Pam's machine
  data_dir = '../func_data/' # where input/subject data is stored
  mask_dir = '../mask' # where mask fils is stored
  dir_out = '../output' # "Conditions" and "Contrast/Conditions/GMSocial_to_GMControl" folders within (other would be Contrast/Pheno/Social/age )
  upper_limit_n_subjs = 5 # max subjects per group for memory/time purposes on local
  all_task_names = [ 'GMControl', 'GMSocial', 'Inscapes']
  all_task_folder_names = ['GMControl', 'GMSocial', 'Inscapes']
  all_task_data_file_names = ['sub-*_GM_NemoC_preproc.nii.gz', 'sub-*_GM_NemoS_preproc.nii.gz', 'sub-*_GM_Inscapes_preproc.nii.gz'] # Rivanna or local
  atlas = mask_dir + "/Yeo2011_7Networks_MNI152_FreeSurferConformed1mm.nii.gz"
  mask_file = mask_dir + "/MNI152_mask.nii.gz"

if not os.path.exists( dir_out ):
  os.makedirs( dir_out )
  print('Created directory: %s' % dir_out)

# END SYSTEM SETUP ------------------------------

# VARIABLE DECS ---------------------------------

# group_assignment_dict = {task_name : i for i, task_name in enumerate(all_task_names)}
n_regions = 268 # number of regions in atlas (shen - 268)

# loading mask 
brain_mask = load_boolean_mask(mask_file)
coords = np.where(brain_mask)
masker_ho = NiftiLabelsMasker(labels_img=atlas)

# load the functional data 
ptp_ids = {} # dictionary to store task_name (key) to an array of all the ptp ids in that task
fnames = {} # stores file names by task
images = {} # stores images by task
masked_images = {} # data images from each subject
bold = {} # Z Scores of trimmed dataset for each task
# group_assignment = [] # TODO: remove references to this if actually not needed
n_subjs = {} # quantity of subjects organized by task
bold_ho = {} # stores z-scored data for each participant

# END VARIABLE DECS -----------------------------

# for each task, get the file names and load the images
for taskIndex in range( len(all_task_names) ):
  print("TASK: " + all_task_names[taskIndex])
  # iterative data prep
  task = all_task_names[taskIndex]
  fnames[task] = get_func_file_names( data_dir, 
                                      task, 
                                      all_task_folder_names[taskIndex], 
                                      all_task_data_file_names[taskIndex],
                                      upper_limit_n_subjs,
                                      verbose = False ) 
  ptp_ids[task] = find_ptp_ids( fnames[task] )
  images[task] = load_images(fnames[task])
  masked_images[task] = mask_images( images[task], brain_mask )
  orig_data = MaskedMultiSubjectData.from_masked_images( masked_images[task], len( fnames[task] ) )
  bold[task] = zscore(orig_data, axis=0)
  n_subj_this_task = np.shape(bold[task])[-1]
  # group_assignment += list(
  #   np.repeat(group_assignment_dict[task], n_subj_this_task)
  # )
  n_TRs = np.shape(orig_data)[0]
  n_subjs[task] = np.shape(bold[task])[-1]
  # Concatenate all of the masked images across participants into a single TR x voxel x subject array
  # TR is time (we take it every 800 ms)
  bold_ho[task] = np.zeros((n_TRs, n_regions, n_subjs[task]))
  # Collect all data 
  row_has_nan = np.zeros(shape=(n_regions,), dtype=bool)
  roi_names = [] # TODO: set this programatically for when running with ROIs

  for ptp in range(n_subjs[task]):
    # get the data for task t, subject s 
    nii_t_s = nib.load(fnames[task][ptp])
    m_raw = masker_ho.fit_transform(nii_t_s)
    bold_ho[task][:,:,ptp] = m_raw
    if (HEADERS):
      m_raw.insert(0, roi_names)
    if (TRIMMED):
      m_raw = m_raw[18:-1, ...]

    f_out = dir_out + task + '/' + str(ptp_ids[task][ptp]) + '_' + task + '_raw.csv'
    np.savetxt(f_out, m_raw, delimiter=",")
  
    # figure out missing rois # TODO: is this needed?
    row_has_nan_ = np.any(np.isnan(bold_ho[task][:,:,ptp]), axis=0)
    if(np.any(row_has_nan_)):
      print("NaN found in participant with ID of: {ptp}")
    row_has_nan[row_has_nan_] = True                

  roi_select = np.logical_not(row_has_nan)
  n_roi_select = np.sum(roi_select)

print("Done.")
