'''
This script contains a function which looks for the best couple (ATP_time, ATP_duration) saved in the folder "savings".
It is used at the beginning of the script "BayesOpt_impulse_calibration.py".
'''

import glob 
import numpy as np
import os

def find_best_savings():
    best_mse1 = 1
    best_mse2 = 1
    best_mse3 = 1


    savings_path = r".." + os.sep + "savings" + os.sep + "*"

    for path in glob.glob(savings_path):
        for subpath in glob.glob(path + os.sep + "*"):
            if subpath.split(os.sep)[-1].startswith('patient_1') & subpath.split(os.sep)[-1].endswith('npy'):
                data = np.load(subpath, allow_pickle=True)
                data = [float(data[0]), float(data[1])]
                if best_mse1 > data[1]:
                    best_patient1 = data[0]
                    best_mse1 = data[1]

            if subpath.split(os.sep)[-1].startswith('patient_2') & subpath.split(os.sep)[-1].endswith('npy'):
                data = np.load(subpath, allow_pickle=True)
                data = [float(data[0]), float(data[1])]
                if best_mse2 > data[1]:
                    best_patient2 = data[0]
                    best_mse2 = data[1]

            if subpath.split(os.sep)[-1].startswith('patient_3') & subpath.split(os.sep)[-1].endswith('npy'):
                data = np.load(subpath, allow_pickle=True)
                data = [float(data[0]), float(data[1])]
                if best_mse3 > data[1]:
                    best_patient3 = data[0]
                    best_mse3 = data[1]
    print("Best savings: ")
    print("Patient1")
    print("nu_2: " + str(best_patient1) + ", MSE:" + str(best_mse1))
    print("Patient2")
    print("nu_2: " + str(best_patient2) + ", MSE:" + str(best_mse2))
    print("Patient3")
    print("nu_2: " + str(best_patient3) + ", MSE:" + str(best_mse3))
    return [[best_patient1, best_mse1], [best_patient2, best_mse2], [best_patient3, best_mse3]]

if __name__ == "__main__":
    find_best_savings()