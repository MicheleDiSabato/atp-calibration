'''
This script is used to compute the numerical ECG to calibrate nu_2. Indeed, the input of the function is the nu_2 parameter 
(called "nu", for simplicity).
'''


# Import libraries for simulation
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

def solver(nu):

    # inner functions for differentiation
    def make_kernel(a):
      """Transform a 2D array into a convolution kernel"""
      a = np.asarray(a)
      a = a.reshape(list(a.shape) + [1,1])
      return tf.constant(a, dtype=1)

    def simple_conv(x, k):
      """A simplified 2D convolution operation"""
      x = tf.expand_dims(tf.expand_dims(x, 0), -1)
      y = tf.nn.depthwise_conv2d(x, k, [1, 1, 1, 1], padding='SAME')
      return y[0, :, :, 0]

    def laplace(x):
      """Compute the 2D laplacian of an array"""
      laplace_iso_k = make_kernel([[0.25, 0.5, 0.25],
                               [0.5, -3., 0.5],
                               [0.25, 0.5, 0.25]])
      laplace_k = make_kernel([[0.0, 1.0, 0.0],
                               [1.0, -4.0, 1.0],
                               [0.0, 1.0, 0.0]])
      return simple_conv(x, laplace_iso_k)

    def laplace_fiber(x):
      """Compute the 2D laplacian of an array"""
      laplace_fib_k = make_kernel([[0.0, 1.0, 0.0],
                               [0, -2., 0.0],
                               [0.0, 1.0, 0.0]])
      return simple_conv(x, laplace_fib_k)

    def diff_y(x):
      """Compute the 2D laplacian of an array"""
      diff_k = make_kernel([[0.0, 0.5, 0.0],
                               [0, 0.0, 0.0],
                               [0.0, -0.5, 0.0]])
      return simple_conv(x, diff_k)

    def diff_x(x):
      """Compute the 2D laplacian of an array"""
      diff_k = make_kernel([[0.0, 0.0, 0.0],
                               [-0.5, 0.0, 0.5],
                               [0.0, 0.0, 0.0]])
      return simple_conv(x, diff_k)


    N = np.int32(128) 
    M = np.int32(64) 
    h = 2/N 

    delta_t = tf.constant(0.01,dtype=tf.float32, shape=()) 
    max_iter_time =  np.int32(450/delta_t)+1 
    scaling_factor = np.int32(1.0/delta_t)

    Trigger = 322 #[ms]
    S2 = Trigger/delta_t  
    save_flag = True

    # initialization
    signals = np.zeros([1,3,np.int32(max_iter_time/scaling_factor)+1], dtype=np.float32)

    # timing of shock
    ICD_time      = np.int32(500/delta_t) 
    # duration of the shock
    ICD_duration  = 5
    # amplitude of the shock
    ICD_amplitude = 1.0

    # Initial Condition
    ut_init   = np.zeros([N, M], dtype=np.float32)
    Iapp_IC   = np.zeros([N, M], dtype=np.float32)
    Iapp_init = np.zeros([N, M], dtype=np.float32)
    Iapp_ICD  = np.zeros([N, M], dtype=np.float32)
    r_coeff = 1.2 + np.zeros([N, M], dtype=np.float32)

    distance_matrix_1 = np.zeros([N, M], dtype=np.float32) # 0.05+1.95*np.random.rand(N, N)
    distance_matrix_2 = np.zeros([N, M], dtype=np.float32) # 0.05+1.95*np.random.rand(N, N)
    for i in range(N): 
        for j in range(M):
            distance_matrix_1[i,j] = 1/(np.sqrt( (i*h-1)**2  + (j*h-1.25)**2))
            distance_matrix_2[i,j] = 1/(np.sqrt( (i*h-1)**2  + (j*h+0.25)**2))

    Iapp_init[np.int32(N/2-0.4/h):np.int32(N/2+0.4/h),np.int32(M/2-0.15/h):np.int32(M/2+0.15/h)] = 1.0
    Iapp_ICD[0:np.int32(0.25/h),np.int32(0.5/h):np.int32(0.65/h)] = 1.0
    Iapp_ICD[N-np.int32(0.25/h):N-1,np.int32(0.5/h):np.int32(0.65/h)] = 1.0


    Iapp_IC[:,0:np.int32(0.05/h)] = 100.0
    Ulist = []

    # physical coefficients
    nu_0 = tf.constant(1.5,dtype=tf.float32, shape=())
    nu_1 = tf.constant(4.4,dtype=tf.float32, shape=())
    nu_2 = tf.constant(nu,dtype=tf.float32, shape=())
    nu_3 = tf.constant(1.0,dtype=tf.float32, shape=())
    v_th = tf.constant(13,dtype=tf.float32, shape=())
    v_pk = tf.constant(100,dtype=tf.float32, shape=())
    D_1 = tf.constant(0.003/(h**2),dtype=tf.float32, shape=())
    D_2 = tf.constant(0.000315/(h**2),dtype=tf.float32, shape=())

    # Create variables for simulation
    Ut   = tf.Variable(ut_init) 
    Wt   = tf.Variable(0*ut_init) 
    Iapp = tf.Variable(Iapp_init)
    IappICD = tf.Variable(Iapp_ICD) 
    IappIC = tf.Variable(Iapp_IC) 
    Dr = tf.Variable(r_coeff, dtype=np.float32)

    Ulist.append(Ut)

    # time advancing
    for i in tqdm(range(max_iter_time)):

        if ((i > -1) & (i < 1+np.int32(2/delta_t)) ) | \
            ((i > np.int32(200/delta_t)) & (i < np.int32(202/delta_t)) ) :
            #coeff_init = ((128/N)**2)*10.0*0.02/delta_t
            coeff_init = 10.0
        else:
            coeff_init = 0.0

        if (i > S2) & (i < S2+np.int32(2/delta_t)):
            #coeff = ((128/N)**2)*100.0*0.02/delta_t
            coeff = 100.0
        else:
            coeff = 0.0

        # nonlinear terms
        I_ion = nu_0*Ut*(1.0-Ut/v_th)*(1.0-Ut/v_pk) + nu_1*Wt*Ut 
        g_ion = nu_2*(Ut/v_pk-nu_3*Wt) 

        # update the solution
        Ut = Ut + delta_t * (  Dr*D_2 * laplace(Ut) + Dr*(D_1-D_2)*laplace_fiber(Ut)\
                - I_ion + coeff_init*IappIC + coeff*Iapp)
        Wt = Wt + delta_t *  g_ion

        # ghost nodes
        tmp_u = Ut.numpy()

        tmp_u[0,:] = tmp_u[2,:]
        tmp_u[N-1,:]  = tmp_u[N-3,:]
        tmp_u[:,0]    = tmp_u[:,2]
        tmp_u[:,M-1]  = tmp_u[:,M-3]

        Ut = tf.Variable(tmp_u)

        Ulist.append(Ut)

        k = np.int32(i/scaling_factor)
        if (np.mod(i,scaling_factor)==0):
            ref = Ulist[i][np.int32(N/2)][np.int32(M/2)]

            # pseudo ECG
            signals[0,0,k] = 1/(h**2)*np.sum(diff_x(Ulist[i][:][:])*diff_y(distance_matrix_1) + diff_y(Ulist[i][:][:])*diff_y(distance_matrix_1)) \
                                    -1/(h**2)*np.sum(diff_x(Ulist[i][:][:])*diff_y(distance_matrix_2) + diff_y(Ulist[i][:][:])*diff_y(distance_matrix_2))

            signals[0,1,k] = i*delta_t

    signals[0,0,:] = signals[0,0,:]/np.amax(signals[0,0,:])

    return signals

