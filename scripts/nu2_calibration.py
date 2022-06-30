'''
This script is run to tune hyperparameter \nu_2 using an "Adaptive" Random Search.

1) denoising of real data
2) tune \nu_2 for each patient and save the values in a dedicated directory

'''

# import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import TF2D_nu_2
import os
from datetime import datetime
from scipy.signal import butter, lfilter, filtfilt
from find_best_savings import find_best_savings
import random
random.seed(1234)

# bandpass filter function:
def bandpass_filter(data, lowcut, highcut, signal_freq, filter_order):
    """
    Method responsible for creating and applying Butterworth filter.
    :param deque data: raw data
    :param float lowcut: filter lowcut frequency value
    :param float highcut: filter highcut frequency value
    :param int signal_freq: signal frequency in samples per second (Hz)
    :param int filter_order: filter order
    :return array: filtered data
    """
    nyquist_freq = 0.5 * signal_freq
    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq
    b, a = butter(filter_order, [low, high], btype="band")
    y = lfilter(b, a, data)
    return y

dataset_p = np.load('signals_3_patients.npy')

# set the parameters for the bandpass filter
sampling_frequency = 1000 # the ecgs are 800ms long and have dimension 800,
                          # so this means the sampling frequency is 1000 sampling/seconds
lowcut_freq = 0.02
highcut_freq = 100  # we need to remove in particular the  high frequency noise
filter_order = 3

exact_denoised = []  # list of the denoised ecgs

exact_denoised.append(bandpass_filter(dataset_p[0,0,0:451], lowcut_freq, highcut_freq, sampling_frequency, filter_order))
exact_denoised.append(bandpass_filter(dataset_p[1,0,0:451], lowcut_freq, highcut_freq, sampling_frequency, filter_order))
exact_denoised.append(bandpass_filter(dataset_p[2,0,0:451], lowcut_freq, highcut_freq, sampling_frequency, filter_order))


# creation of the savings directory
exps_dir = ".."+ os.sep +"savings"
if not os.path.exists(exps_dir):
    os.makedirs(exps_dir)

now = datetime.now().strftime('%b%d_%H-%M-%S')

for ind_s in range(3):
    def fun_norm(x):
        # compute the numerical solution
        signals = TF2D_nu_2.solver(x)
        # create the saving directory if not already done
        exp_dir = os.path.join(exps_dir, str(now))
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
        savings_directory = os.path.join(exp_dir, 'signal_patient_' + str(ind_s + 1) + 'nu2_' + str(x) + '.npy')
        # save the obtained numerical solution in order to keep track of the previous results computed by
        # different runs of the code
        np.save(savings_directory, signals)
        # compute the Mean square error of the numerical solution w.r.t. the exact one
        MSE = mean_squared_error(signals[0, 0, :], exact_denoised[ind_s])

        # plot the numerical solution and the exact one in order to check the results
        fig, axs = plt.subplots(figsize=(20, 4))
        axs.plot(exact_denoised[ind_s], label='exact (denoised) solution', color='blue')
        axs.plot(signals[0, 0, :], label='numerical solution', color='red')
        axs.set_title("Patient: " + str(ind_s + 1) + ", Nu_2 = " + str(x) + ", MSE = " + str(MSE))
        axs.legend()
        plt.savefig(exp_dir + os.sep +'patient_' + str(ind_s + 1) + 'nu_2_' + str(x) + 'mse_' + str(MSE) + '.png')
        np.save(exp_dir + os.sep + 'patient_' + str(ind_s + 1) + 'nu_2_' + str(x) + 'mse_' + str(MSE), np.array([x, MSE]))

        return MSE

    # Adaptive Random Search with 20 iteration per patient:
    # the first 10 searching steps are performed starting from the best guess and adding a
    # gaussian noise with a fixed variance (which is the "vanilla" gaussian random search), 
    # while in the last 10 we added a gaussian noise with variance linearly decreasing with the number of iterations.
    # The reasoning behind this approach is that as more iterations are performed, the algorithm will be closer
    # to the best nu_2, so the "jump" from the best current nu_2 to the next guess can be moderate.

    nu_2_est = 0.012
    norm_est = fun_norm(nu_2_est)
    it = 0
    print('Iteration ' + str(it + 1))
    print('Guess %f MSE %f ' % (nu_2_est, norm_est))
    it = it+1
    it_max = 20

    while it < it_max:
        flag = True
        # new sample:
        if it < 10:
            while flag:
                nu_2_guess = nu_2_est + 0.0004 * np.random.normal(0,1, size=1)
                if nu_2_guess < 0.0116 or nu_2_guess > 0.0124:
                    flag = True
                else:
                    flag = False
        else:
            # decreasing variance of the uniform random variable
            while flag:
                nu_2_guess = nu_2_est + 0.0004 * np.random.normal(0,1, size=1) * (it_max - it)/(it_max - 10)
                if nu_2_guess < 0.0116 or nu_2_guess > 0.0124:
                    flag = True
                else:
                    flag = False

        norm = fun_norm(nu_2_guess)
        if norm < norm_est:
            norm_est = norm
            nu_2_est = nu_2_guess
        print('Iteration ' + str(it+1))
        print('Guess %f MSE %f ' % (nu_2_guess, norm))
        it += 1


    print('Final estimate patient ' + str(ind_s+1) + ': %f, MSE: %f' % (nu_2_est, norm_est))
