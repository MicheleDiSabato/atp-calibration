# atp-calibration

## Theoretical framework:
The framework for this project is the following: the normal diffusion of the potential in the heart is hindered by a [re-entry](https://www.youtube.com/watch?v=yLI4yj1TZhc) of the signal, possibly caused by scar tissue in specific areas of the heart. This problem leads to some complications, which could be fatal. In order to restore the normal diffusion of the potential through the affectted region, an Anti-Tachycardia Pacing (ATP) device is inserted: its purpose is to deliver an impulse to avoid the re-entry. The role of the impulse is to reset the ECG and the device checks the effectiveess of the shock tracking the ECG in a certain time interval after the shock. This period is called *tracking window* and it lasts from 600 to 800 milliseconds.

We are given 
* three ECG signal observed for some milliseconds of three different patients;
* a numerical solver to compute an approximation of an ECG to solve the [monodomain problem](https://en.wikipedia.org/wiki/Monodomain_model), couple with the [Rogers-McCulloch model](https://ieeexplore.ieee.org/document/310090?reload=true) for the ionic current.

## Goals:
We want to:
* calibrate patient-specific parameter ν<sub>2</sub> for the Rogers-McCulloch model;
* find the *best* values for the timing (t<sup>*</sup>) and duration (Δt<sup>*</sup>) of the impulse delivered by the ATP;  

We quantify the effectiveness of the pair (t<sup>'</sup>, Δt<sup>'</sup>) in terms of the ability of the resulting impuls to :
1. annihilate the ECG in the tracking window 
2. preserve the device's battery as long as possible by minimizing the duration of the impulse 


## Calibration for ν<sub>2</sub>:

+ risultati
+ grafici

## Calibration for t and delta_t
relazione tra delta t e norma L2 della loss 

+ risultati 
+ grafici

# 
