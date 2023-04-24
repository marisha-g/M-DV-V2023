"""
Import libraries
"""
import os
import time
from itertools import islice
import warnings
warnings.simplefilter("ignore")
from tqdm.notebook import tqdm

import numpy as np
import pandas as pd

# Tensorflow
import tensorflow as tf
import tensorflow_probability as tfp

# Shortcuts
tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tfp.math.psd_kernels

from anisotropic_kernel import GeometricAnisotropy

def kernel_setup(mode, tfk_kernel, vals_for_mean=None, mean_func=None): 
    """
    Define kernel function to be used in the Gaussian Process.
    Mode: anisotropy or isotropy
    kernel: choose base kernel, either exponentiaded quadratic or linear tensorflow tfp.math.psd_kernels module
    mean func: choose if mean function should be constant 0 (Default) or constant target mean (1)
    """

    # Constrain to make sure certain parameters are strictly positive
    constrain_positive = tfb.Shift(np.finfo(np.float64).tiny)(tfb.Exp())

    # Noise variance of observations
    observation_noise_variance_var = tfp.util.TransformedVariable(
        initial_value=0.001, bijector=constrain_positive, dtype=np.float64,
        name='observation_noise_variance')
    
    # Mean function
    if mean_func == "yes":
        observations_mean = tf.constant([np.mean(vals_for_mean)], dtype=tf.float64)
        mean_fn = lambda _: observations_mean
    else:
        mean_fn = None

    # Define kernel hyperparameters
    amplitude_var = tfp.util.TransformedVariable(
        initial_value=1., 
        dtype=np.float64, 
        bijector=constrain_positive,
        name='amplitude')
    
    angle_var = tfp.util.TransformedVariable(
        initial_value=np.pi / 2, 
        bijector=constrain_positive, 
        dtype=tf.float64, 
        name="angle")

    lambda1_var = tfp.util.TransformedVariable(
        initial_value=1., 
        dtype=np.float64, 
        bijector=constrain_positive,
        name='lambda1')

    lambda2_var = tfp.util.TransformedVariable(
        initial_value=1., 
        dtype=np.float64, 
        bijector=constrain_positive,
        name='lambda2')
    
    length_scale_var = tfp.util.TransformedVariable(
        initial_value=10., 
        dtype=np.float64, 
        bijector=constrain_positive,
        name='length_scale')

    bias_amplitude_var = tfp.util.TransformedVariable(
        initial_value=0.0001, 
        dtype=np.float64, 
        bijector=constrain_positive,
        name='bias_amplitude')

    slope_amplitude_var = tfp.util.TransformedVariable(
        initial_value=1., 
        dtype=np.float64, 
        bijector=constrain_positive,
        name='slope_amplitude')

    shift_var = tfp.util.TransformedVariable(
        initial_value=0.0001, 
        dtype=np.float64, 
        bijector=constrain_positive,
        name='shift')
    
    if mode == "anisotropy":

        if tfk_kernel == "rbf":
            base_kernel = tfk.ExponentiatedQuadratic(amplitude_var)
            kernel = GeometricAnisotropy(base_kernel, lambda1_var, lambda2_var, angle_var)
            trainable_variables = [v.variables[0] for v in [
                amplitude_var, 
                lambda1_var,
                lambda2_var,
                angle_var,
                observation_noise_variance_var
                ]]
            variables = [amplitude_var, lambda1_var, lambda2_var, angle_var, observation_noise_variance_var]
        
        if tfk_kernel == "matern":
            base_kernel = tfk.MaternThreeHalves(amplitude_var)
            kernel = GeometricAnisotropy(base_kernel, lambda1_var, lambda2_var, angle_var)
            trainable_variables = [v.variables[0] for v in [
                amplitude_var, 
                lambda1_var,
                lambda2_var,
                angle_var,
                observation_noise_variance_var
                ]]
            variables = [amplitude_var, lambda1_var, lambda2_var, angle_var, observation_noise_variance_var]
        
        if tfk_kernel == "combined":
            base_kernel1 = tfk.Linear(bias_amplitude_var, slope_amplitude_var, shift_var)
            base_kernel2 = tfk.ExponentiatedQuadratic(amplitude_var)
            base_kernel3 = GeometricAnisotropy(base_kernel2, lambda1_var, lambda2_var, angle_var)
            kernel = (base_kernel1 + base_kernel3)
            trainable_variables = [v.variables[0] for v in [
                bias_amplitude_var, 
                slope_amplitude_var, 
                shift_var,
                amplitude_var, 
                lambda1_var,
                lambda2_var,
                angle_var,
                observation_noise_variance_var
                ]]
            variables = [
                bias_amplitude_var, slope_amplitude_var, shift_var, 
                amplitude_var, lambda1_var, lambda2_var, angle_var,
                observation_noise_variance_var
                ]

    if mode == "isotropy":

        if tfk_kernel == "rbf":
            kernel = tfk.ExponentiatedQuadratic(amplitude_var, length_scale_var)
            trainable_variables = [v.variables[0] for v in [
                amplitude_var, 
                length_scale_var,
                observation_noise_variance_var
                ]]
            variables = [amplitude_var, length_scale_var, observation_noise_variance_var]
        
        if tfk_kernel == "matern":
            kernel = tfk.MaternThreeHalves(amplitude_var, length_scale_var)
            trainable_variables = [v.variables[0] for v in [
                amplitude_var, 
                length_scale_var,
                observation_noise_variance_var
                ]]
            variables = [amplitude_var, length_scale_var, observation_noise_variance_var]

        if tfk_kernel == "combined":
            base_kernel1 = tfk.Linear(bias_amplitude_var, slope_amplitude_var, shift_var)
            base_kernel2 = tfk.ExponentiatedQuadratic(amplitude_var, length_scale_var)
            kernel = (base_kernel1 + base_kernel2)
            trainable_variables = [v.variables[0] for v in [
                bias_amplitude_var, 
                slope_amplitude_var, 
                shift_var,
                amplitude_var, 
                length_scale_var,
                observation_noise_variance_var
                ]]
            variables = [
                bias_amplitude_var, slope_amplitude_var, shift_var, 
                amplitude_var, length_scale_var, observation_noise_variance_var
                ]

    return kernel, mean_fn, trainable_variables, observation_noise_variance_var, variables