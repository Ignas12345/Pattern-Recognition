import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from GaussD import GaussD
import numpy as np
from MarkovChain import MarkovChain
from HMM import HMM

def initialize_emissions(activity_data):
    standing_mean = activity_data[0]['Absolute acceleration (m/s^2)'].mean()
    standing_std = activity_data[0]['Absolute acceleration (m/s^2)'].std()
    
    walking_mean = activity_data[1]['Absolute acceleration (m/s^2)'].mean()
    walking_std = activity_data[1]['Absolute acceleration (m/s^2)'].std()
    
    running_mean = activity_data[2]['Absolute acceleration (m/s^2)'].mean()
    running_std = activity_data[2]['Absolute acceleration (m/s^2)'].std()

    standing_dist = [standing_mean, standing_std]
    walking_dist = [walking_mean, walking_std]
    running_dist = [running_mean, running_std]
    return standing_dist, walking_dist, running_dist
    

file_path_running = 'Acceleration_without_g_running.xls'
file_path_standing = 'Acceleration_without_g_standing.xls'
file_path_walking = 'Acceleration_without_g_walking.xls'

# Load the data from Excel files
running_data = pd.read_excel(file_path_running, engine='xlrd', usecols=["Absolute acceleration (m/s^2)"])
standing_data = pd.read_excel(file_path_standing, engine='xlrd', usecols=["Absolute acceleration (m/s^2)"])
walking_data = pd.read_excel(file_path_walking, engine='xlrd', usecols=["Absolute acceleration (m/s^2)"])

# Display the first few rows of the Data to check the correct column is loaded
"""print("Standing Data Head:", standing_data.head())
print("Walking Data Head:", walking_data.head())
print("Running Data Head:", running_data.head())"""


running_train, running_test = train_test_split(running_data, test_size=0.2, random_state=42)
standing_train, standing_test = train_test_split(standing_data, test_size=0.2, random_state=42)
walking_train, walking_test = train_test_split(walking_data, test_size=0.2, random_state=42)
# Print the shapes of the train and test sets to verify
"""print("running_data Size:", running_data.shape)
print("Running Train Size:", running_train.shape)
print("Running Test Size:", running_test.shape)
print("Standing Train Size:", standing_train.shape)
print("Standing Test Size:", standing_test.shape)
print("Walking Train Size:", walking_train.shape)
print("Walking Test Size:", walking_test.shape)"""
#Initial guesses on the distributions


q = np.array([1/3, 1/3, 1/3]) 
A = np.array([
    [0.9, 0.05, 0.05],
    [0.05, 0.9, 0.05],
    [0.05, 0.05, 0.9]
])
chain = MarkovChain(q, A)
training_data = [standing_train, walking_train, running_train]
initial_standing_distribution, initial_walking_distribution, initial_running_distribution = initialize_emissions(training_data)
"""print(initial_standing_distribution)
print(initial_walking_distribution)
print(initial_running_distribution)"""
standing_distribution = GaussD(means=[initial_standing_distribution[0]], stdevs=[initial_standing_distribution[1]])
walking_distribution = GaussD(means=[initial_walking_distribution[0]], stdevs=[initial_walking_distribution[1]])
running_distribution = GaussD(means=[initial_running_distribution[0]], stdevs=[initial_running_distribution[1]])
h = HMM(chain, [standing_distribution, walking_distribution, running_distribution])

combined_training_array = np.concatenate([
    standing_train['Absolute acceleration (m/s^2)'].values,
    walking_train['Absolute acceleration (m/s^2)'].values,
    running_train['Absolute acceleration (m/s^2)'].values
])

"""print("combined_training_array", combined_training_array)
print("combined_training_array shape ", combined_training_array.shape)"""

nStates = 3
nSamples = len(combined_training_array)
pX = np.zeros((nStates, nSamples))
scale_factors = np.zeros(nSamples)
for t in range(nSamples):
    for j, g in enumerate([standing_distribution, walking_distribution, running_distribution]):
        pX[j, t] = g.prob(combined_training_array[t])
    scale_factors[t] = pX[:, t].max()
    pX[:, t] /= scale_factors[t]
alpha_hat, c = chain.forward(pX)
beta_hat = chain.backward(c, pX)
print(h.stateGen.A)
h.updateA(alpha_hat, beta_hat, pX)
print(h.stateGen.A)
