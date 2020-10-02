import numpy as np
import pandas as pd

'''
Takes in a 2D array-like object (numpy definition) and saves it as a csv to a provided path.
as_int parameter forces the datatype to be an integer in the csv.
'''
def save_arraylike_as_csv(data, path, as_int=False):
  if(as_int):
    np.savetxt(path, data, delimiter=",", fmt='%i')
  else: 
    np.savetxt(path, data, delimiter=",")

"""
From a given path, reads a csv and returns either a Pandas dataframe or Numpy array
"""
def read_csv_matrix(path, as_dataframe=False):
  if(as_dataframe):
    return pd.read_csv(path)  
  else:
    return np.genfromtxt(path, delimiter=',')
