# atp-calibration

## Theoretical framework:
The framework for this project is the following: the normal diffusion of the potential in the heart is hindered by a [re-entry](https://www.youtube.com/watch?v=yLI4yj1TZhc) of the signal, possibly caused by scar tissue in specific areas of the heart. This problem leads to some complications, which could be fatal. In order to restore the normal diffusion of the potential through the affectted region, an Anti-Tachycardia Pacing (ATP) device is inserted: its purpose is to deliver an impulse to avoid the re-entry. The role of the impulse is to reset the ECG and the device checks the effectiveess of the shock tracking the ECG in a certain time interval after the shock. This period is called *tracking window* and it lasts from 600 to 800 milliseconds.

We are given 
* three ECG signal observed for some milliseconds of three different patients;
* a numerical solver to compute an approximation of an ECG to solve the [monodomain problem](https://en.wikipedia.org/wiki/Monodomain_model), couple with the [Rogers-McCulloch model](https://ieeexplore.ieee.org/document/310090?reload=true) for the ionic current.

## Goals:
We want to:
* calibrate patient-specific parameter ν<sub>2</sub> for the Rogers-McCulloch model;
* find the *best* values for the timing (t<sup>best</sup>) and duration (Δt<sup>best</sup>) of the impulse delivered by the ATP;  

We quantify the effectiveness of pair (t<sup></sup>, Δt<sup></sup>) in terms of the ability of the resulting impulse to:
1. annihilate the ECG in the tracking window 
2. preserve the device's battery as long as possible by minimizing the duration of the impulse 

## Calibration for ν<sub>2</sub>:
Before working on the ATP device, we need to tune hyperparameter ν<sub>2</sub>. To do so we used a modifed version of a naive random search algorthm, which we call *Adaptive Random Search* (see page 2, Section 2.1 of our [report](report.pdf)).

**Remarks:**
1. The ECG provided are noisy: they were first denoised using a low-pass bandpass filter called [Butterworth filter](https://en.wikipedia.org/wiki/Butterworth_filter). The results are shown in the following plots:
| Patient 1 |  Patient 2 |  Patient 3 |
:-------------------------:|:-------------------------:|:-------------------------:
![](readme_images/patient_1_noise.png)  | ![](readme_images/patient_2_noise.png) | ![](readme_images/patient_3_noise.png)

2. For each iteration, the sampling of ν<sub>2</sub> is repeated until it falls within its bounds.
3. The variance of the Gaussian from which we sample ν<sub>2</sub> decreases according to the number of iterations.
4. We used 20 iterations: for each iteration we need to compute the solution of the monodomain problem coupled with the model for the ionic current, which is quite costly.
5. 
## Calibration for t and delta_t
relazione tra delta t e norma L2 della loss 

+ risultati 
+ grafici

# 
