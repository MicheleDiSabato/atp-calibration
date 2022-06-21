# atp-calibration

<!-- <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script> -->

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
![p1](readme_images/patient_1_noise.png)  | ![p2](readme_images/patient_2_noise.png) | ![p3](readme_images/patient_3_noise.png)

2. For each iteration, the sampling of ν<sub>2</sub> is repeated until it falls within its bounds.
3. The variance of the Gaussian from which we sample ν<sub>2</sub> decreases according to the number of iterations.
4. We used 20 iterations: for each iteration we need to compute the solution of the monodomain problem coupled with the model for the ionic current, which is quite costly.

## Calibration for (t<sup>best</sup>, Δt<sup>best</sup>)
Calibrating t<sup></sup> and Δt<sup></sup> using [grid search](https://towardsdatascience.com/grid-search-for-model-tuning-3319b259367e) or even our own version of a Naive Random Search (which hwe called Adaptive Random Search) is too costly, since for each possible couple (t<sup></sup>, Δt<sup></sup>) we would need to solve the monodomain problem for each patient. 

To avoid wasting time in useless evaluations, we relied on [Bayesian Otimization](https://arxiv.org/abs/1807.02811), in particular on [this](https://github.com/fmfn/BayesianOptimization) opensource python library, which needs to be installed to run the code contained in this repository:

```
pip install bayesian-optimization
```

The Bayesian Optimization algorithm is specifically designed to minimize the number of evaluations of the objective function, which in our case is the L2 norm of the ECG signal of each patient in the tracking window (600, 800) milliseconds. This approach is particularly useful when the objective function is expensive to be evaluated, as in our case.

**Remarks:**
1. Bayesian Optimization is implemented to solve a **maximization** problem, so to use this library for our purposes[^1], we used the opposite of the L2 norm of the numerical ECG, hence the "minus" sign in the objective function.
2. To take into account also the battery duration, we decided to add to the loss function a term which is proportional to the square of the duration of the impulse, to try to be more conservative with respect to the battery consumption.
3. The ATP device comes into play only after 450 milliseconds, therefore the ECG until the 450-th millisecond is always the same. Still, the numerical solver (which relies on an iterative procedure) needs to compute the solution from the very beginning. To avoid wasting time on simulating the first 450 milliseconds, we computed the ECG once and for all amd saved it: at each step of the Bayesian Optimization algorithm, we simply load the numerical ECG until the 450-th millisecond and go on from there using the solver. **Please look at [WARNING.md](WARNING.md) for more details.**

### Loss function:
We used two typed of loss function:
1. $||\hat{u}||_{L^2} + \lambda \Delta t$
2. $||\hat{u}||_{L^2} + \lambda (\Delta t)^2$

[^1]: we want to reset the ECG, therefore we are dealing with a minimization task.

where 
* $u$ is the numerical ECG
* $\Delta t$ is the ATP impulse duration
* $\lambda$ is a term which quantifies the fact that higher values of $\Delta t$ should be avoided, since they undemine the device's battery.

We decided to use:

| item | Patient 1 |  Patient 2 |  Patient 3 |
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
|**Loss** | $\|\|hat{u}||_{L^2}+(\Delta t)^2$ | $||\hat{u}||_{L^2}+(\Delta t)^2$ | $||\hat{u}||_{L^2}+0.001\Delta t$
|**\lambda** | 1  | 1 | 0.001




























