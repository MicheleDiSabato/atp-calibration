'''
Plots the best nu_2 found for each patient and plots the associated MSE, which is 
the L2 distance between the (denoised) true ECG and the numerical ECG obtained with that 
specific choice of nu_2. See file nu2_calibration.py for more details on the calibration of nu_2.
'''

import glob 
import numpy as np
import os 
import matplotlib.pyplot  as plt
from find_best_savings import find_best_savings
patient1_results = []
patient2_results = []
patient3_results = []

savings_path = ".." + os.sep + "savings" + os.sep + "*"

for path in glob.glob(savings_path):
    for subpath in glob.glob(path + os.sep + "*"):
        if subpath.split(os.sep + 'patient_')[-1].startswith('1') & subpath.split(os.sep)[-1].endswith('npy'):
            data = np.load(subpath, allow_pickle=True)
            data = [float(data[0]), float(data[1])]
            patient1_results.append(data)

        if subpath.split(os.sep + 'patient_')[-1].startswith('2') & subpath.split(os.sep)[-1].endswith('npy'):
            data = np.load(subpath, allow_pickle=True)
            data = [float(data[0]), float(data[1])]
            patient2_results.append(data)

        if subpath.split(os.sep + 'patient_')[-1].startswith('3') & subpath.split(os.sep)[-1].endswith('npy'):
            data = np.load(subpath, allow_pickle=True)
            data = [float(data[0]), float(data[1])]
            patient3_results.append(data)

patient1_results = np.asarray(patient1_results)
patient2_results = np.asarray(patient2_results)
patient3_results = np.asarray(patient3_results)
find_best_savings()
plt.scatter(patient1_results[:,0], patient1_results[:,1], color = 'blue', marker = 'o', label = 'patient1', s = 30)
plt.scatter(patient2_results[:,0], patient2_results[:,1], color = 'red', marker = 'o', label = 'patient2', s = 30)
plt.scatter(patient3_results[:,0], patient3_results[:,1], color = 'black',  marker = 'o', label = 'patient3', s = 30)
plt.xlabel("nu_2")
plt.ylabel("mse")
plt.legend()
plt.show()

