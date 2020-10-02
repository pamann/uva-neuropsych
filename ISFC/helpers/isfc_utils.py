import os
import numpy as np
import csv
import pandas as pd
import re
from collections import Counter

def select_ptps(in_path, fname_template, ptp_id_list, quiet=True):
  f = open(in_path, "r")
  ptp_id_list = f.read().splitlines()

  # TODO: change this to a real join
  unconf_fnames = [in_path + fname_template for ptp_id in ptp_id_list]
  conf_fnames = []
  missing_fnames = []
  # TODO: extract this and make it less gross
  for fpath in unconf_fnames:
    if(os.path.exists(fpath)):
      conf_fnames.append(fpath)
    else: 
      missing_fnames.append(fpath)
  fnames = conf_fnames
  if not quiet:
    if(len(missing_fnames) > 0):
      print(f"Missing participant files for current task: ")
      print('[%s]' % ', '.join(map(str, missing_fnames)))
    else:
      print(f"No missing participant files for current task.")
  return (fnames, missing_fnames)

def read_shen_networks(csv_path):
  networks = np.genfromtxt(csv_path, delimiter=',', dtype=int)
  node_link = {}
  for node in networks:
    node_link[node[0]] = node[1]
  node_link.pop(-1) # removes header columns (they evaluate to -1 when converted to integer)
  return node_link

# assuming a full matrix, not upper diagonal
def sort_by_networks(node_link, corr_matrix, labels):
  network_index = []
  for i in range(1, len(corr_matrix)+1): # gets the network associated with each node from the map
    network_index.append(node_link[i])

  # convert matrix to pandas df with labels as x and y 
  corr_df = pd.DataFrame(corr_matrix, columns=labels)
  corr_df.insert(loc=0, column="ls", value=labels)
  corr_df.index = network_index # sets row index (aka row labels) to the associated networks

  # sort by column
  corr_df = corr_df.sort_index(axis=0)
  # extract new label order
  reordered_labels = corr_df["ls"].tolist()
  corr_df = corr_df.drop("ls", 1)

  # transpose, repeat (for sorting rows)
  corr_df.T
  corr_df.index = network_index
  corr_df = corr_df.sort_index(axis=0)

  # gets dict of group labels to their counts
  # group_counts = Counter(corr_df.index)
  # total_count = sum(group_counts.values())
  # group_proportions = []
  # for group, count in group_counts.items():
  #   group_proportions.append(count/total_count)
  group_labels = []
  for label in reordered_labels:
    group_labels.append(node_link[label])

  # convert corr_df back to 2D list 
  reordered_matrix = corr_df.values.tolist() # converts dataframe to 2D list row-wise (rows are the sublists)
  
  return (reordered_matrix, network_index, reordered_labels)
   