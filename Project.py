import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from GaussD import GaussD
import numpy as np
file_path = 'Acceleration_without_g_running.xls'
running_data = pd.read_excel(file_path, engine='xlrd')

file_path = 'Acceleration_without_g_standing.xls'
standing_data = pd.read_excel(file_path, engine='xlrd')

file_path = 'Acceleration_without_g_walking.xls'
walking_data = pd.read_excel(file_path, engine='xlrd')

#Display the first few rows of the Data
"""print(standing_data.head())
print(walking_data.head())
print(running_data.head())"""
running_train, running_test = train_test_split(running_data, test_size=0.2, random_state=42)
standing_train, standing_test = train_test_split(standing_data, test_size=0.2, random_state=42)
walking_train, walking_test = train_test_split(walking_data, test_size=0.2, random_state=42)

"""# Print the shapes of the train and test sets to verify
print("running_data Size:", running_data.shape)
print("Running Train Size:", running_train.shape)
print("Running Test Size:", running_test.shape)
print("Standing Train Size:", standing_train.shape)
print("Standing Test Size:", standing_test.shape)
print("Walking Train Size:", walking_train.shape)
print("Walking Test Size:", walking_test.shape)"""
g1 = GaussD(means=[0], stdevs=[1])
g2 = GaussD(means=[1], stdevs=[1])
g3 = GaussD(means=[2], stdevs=[1])
A = np.array([[0.33, 0.33, 0.33], [0.33, 0.33, 0.33], [0.33, 0.33, 0.33]])
nStates = 3
