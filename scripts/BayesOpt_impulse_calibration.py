'''
This is the implementation of a Bayesian approach aimed at finding the best values for the impulse time and duration.
The Bayesian Optimization algorithm is specifically designed to minimize the number of evaluations
of the objective function, which in our case is the L2 norm of the ECG signal of each patient
in the tracking window (600, 800) milliseconds. This approach is particularly useful when
the objective function is expensive to be evaluated, as in our case.
More detail on the Bayesian Optimization library that we used can be found at:
    https://github.com/fmfn/BayesianOptimization

To install the library, please run the following command:

    pip install bayesian-optimization


REMARK: Bayesian Optimization is implemented to solve a maximization problem, so to use this library
for our purposes, we used the opposite of the L2 norm of the numerical ECG, hence the "minus"
sign in the objective function.

REMARK: to take into account also the battery duration, we decided to add to the loss function
a term which is proportional to the square of the duration of the impulse, to try to be more
conservative with respect to the battery consumption.
'''

# Import libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import glob
import os
from datetime import datetime
# BayesOpt libraries
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
# Import useful functions
from find_best_savings import find_best_savings
from solver_ATP import solver_ATP

# Definition of best_nu vector:
fbp = find_best_savings()
best_nu = []
for i in range(3):
    best_nu.append(fbp[i][0])

# Define black_box_functions for Bayesian Optimization:
l = 0.0001 # lambda parameter to take into account ATP_duration in the loss function

def black_box_function_patient_1(x, y):
        '''
        x = ATP_time
        y = ATP_duration
        '''
        import numpy as np

        signals = solver_ATP(ATP_time = x, ATP_duration = y, best_nu = [best_nu[0]], idx = 0, max_time = 800, patients = 1, verbose = 1, Flag = True, starting_time_ecg=449)
        sum = np.linalg.norm(signals[0, 0, 600:])
        out = -sum
        # save the result
        savings_directory=os.path.join(exp_dir, 'signal_patient_1_ATP_time_'+str(x)+'_ATP_dur_'+str(y)+'.npy')
        np.save(savings_directory,signals)

        fig, axs = plt.subplots(figsize=(20, 4))
        # plot and save the figure
        axs.plot(signals[0,1,:],signals[0,0,:])
        axs.set_title("Patient: 1 , ATP_time = " + str(x) + ", ATP_dur = " + str(y))
        plt.savefig(exp_dir + os.sep + 'patient_1_ATP_time_' + str(x) + '_ATP_dur_' + str(y)+ '.png')

        return out - y*y*l

def black_box_function_patient_2(x, y):
        '''
        x = ATP_time
        y = ATP_duration
        '''
        import numpy as np

        signals = solver_ATP(ATP_time = x, ATP_duration = y, best_nu = [best_nu[1]], idx = 1, max_time = 800, patients = 1, verbose = 1, Flag = True, starting_time_ecg=449)
        sum = np.linalg.norm(signals[0, 0, 600:])
        out = -sum
        # save the result
        savings_directory=os.path.join(exp_dir, 'signal_patient_2_ATP_time_'+str(x)+'_ATP_dur_'+str(y)+'.npy')
        np.save(savings_directory,signals)

        fig, axs = plt.subplots(figsize=(20, 4))
        # plot and save the figure
        axs.plot(signals[0,1,:],signals[0,0,:])
        axs.set_title("Patient: 2 , ATP_time = " + str(x) + ", ATP_dur = " + str(y))
        plt.savefig(exp_dir + os.sep + 'patient_2_ATP_time_' + str(x) + '_ATP_dur_' + str(y)+ '.png')

        return out - y*y*l

def black_box_function_patient_3(x, y):
        '''
        x = ATP_time
        y = ATP_duration
        '''
        import numpy as np
        signals = solver_ATP(ATP_time = x, ATP_duration = y, best_nu = [best_nu[2]], idx = 2    , max_time = 800, patients = 1, verbose = 1, Flag = True, starting_time_ecg=449)
        sum = np.linalg.norm(signals[0, 0, 750:])
        out = -sum
        # save the result
        savings_directory=os.path.join(exp_dir, 'signal_patient_3_ATP_time_'+str(x)+'_ATP_dur_'+str(y)+'.npy')
        np.save(savings_directory,signals)

        fig, axs = plt.subplots(figsize=(20, 4))
        # plot and save the figure
        axs.plot(signals[0,1,:],signals[0,0,:])
        axs.set_title("Patient: 3 , ATP_time = " + str(x) + ", ATP_dur = " + str(y))
        plt.savefig(exp_dir + os.sep + 'patient_3_ATP_time_' + str(x) + '_ATP_dur_' + str(y)+ '.png')

        return out - y*y*l

# Define optimizers for Bayesian Optimization:

# REMARK: the optimization bounds were set to 0.1, so that the algorithm
# does not lose time in considering the case ATP_duration = 0

# REMARK: 'x' stands for the impulse time and 'y' for the impulse duration
optimizer1 = BayesianOptimization(
    f=black_box_function_patient_1,
    pbounds={'x': (450.0,525.0), 'y': (0.1,10.0)},
    verbose=2,
    random_state = 123 + 0
)
optimizer2 = BayesianOptimization(
    f=black_box_function_patient_2,
    pbounds={'x': (450.0,525.0), 'y': (0.1,10.0)},
    verbose=2,
    random_state = 123 + 1
)
optimizer3 = BayesianOptimization(
    f=black_box_function_patient_3,
    pbounds={'x': (450.0,525.0), 'y': (1.0 ,4.0)},
    verbose=2,
    random_state = 123 + 2
)

optimizer = [optimizer1, optimizer2, optimizer3]

exps_dir = ".."+ os.sep +"plot"
if not os.path.exists(exps_dir):
    os.makedirs(exps_dir)
now = datetime.now().strftime('%b%d_%H-%M-%S')
exp_dir = os.path.join(exps_dir, str(now))
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)

# 3 steps for 3 patients:
for patient in range(3):
    print("==============================")
    print("elaborating patient ", patient+1)
    print("==============================")

    # 3 steps of pure random exploration
    optimizer[patient].maximize(init_points=7, n_iter=0, acq='ei')
    # 20 steps of Bayesian Optimization; to allow for more exploration we could set
    # kappa to 6 and kappa_decay to 0.8, for example.
    optimizer[patient].maximize(init_points=0, n_iter=50, acq='ei')


# Find the best value:
best_values = []
for patient in range(3):
    print("Best values for patient", patient + 1, ":", optimizer[patient].max)
    best_values.append(optimizer[patient].max)


targets = []
ATP_times = []
ATP_durations = []
for patient in range(3):
    targets.append([res["target"] for res in optimizer[patient].res])
    ATP_times.append([res["params"]['x'] for res in optimizer[patient].res])
    ATP_durations.append([res["params"]['y'] for res in optimizer[patient].res])
    print("===============================")
    print("output of patient", patient + 1)
    print("===============================")
    print("target:", targets[patient])
    print("ATP_time:", ATP_times[patient])
    print("ATP_duration:", ATP_durations[patient])
