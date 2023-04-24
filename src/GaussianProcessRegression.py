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

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

tf.random.set_seed(42)

# Visualization
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import matplotlib.pyplot as plt

#from anisotropic_kernel import GeometricAnisotropy
from anisotropic_kernel import GeometricAnisotropy

# Matrics
import properscoring as ps
from sklearn.metrics import r2_score, mean_squared_error

"""
Load data 
"""
filename = open("C:/Users/Maris/Documents/GitHub/master/data/Grunnboringer_Berum.csv")
data = pd.read_csv(filename, header=0)

"""
Data preprocessing 
"""
# Check and remove rows containing NaN values
data.replace('<Null>', np.nan, inplace = True)
print(" \nCount total NaN at each column in DataFrame: \n\n", data.isnull().sum())
df = data.dropna(subset=['DTB'])
# Cast DTB column to float
df['DTB'] = df['DTB'].astype(np.float64)

# Create new column for relative DTB (terrain height minus DTB) 
#df["ABS_DTB"] = df["TERRAIN"] - df["DTB"]

# Split data to x and y
y = df.DTB.values.reshape(-1, 1)
x = np.array([df.X, df.Y]).T 

# Split dataset into test and training sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

# Scale data with StandardScaler
scalerX = StandardScaler().fit(X_train)
scalery = StandardScaler().fit(y_train)

X_train = scalerX.transform(X_train)
y_train_sc = scalery.transform(y_train)
X_test = scalerX.transform(X_test)
y_test_sc = scalery.transform(y_test)

# Convert y_train and y_test to 1-dimensional arrays
y_train = y_train_sc.ravel()
y_test = y_test_sc.ravel()

"""
Plot data
"""
fig = go.Figure(go.Scatter(mode="markers", x=X_train[:,0], y=X_train[:,1],
                           marker_line_color="midnightblue", marker_color=y_train,
                           marker_line_width=1, marker_size=5))

fig.show()

def kernel_func(mode, tfk_kernel, vals_for_mean=None, mean_func=None): 
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
        if tfk_kernel == "linear":
            base_kernel = tfk.Linear(bias_amplitude_var, slope_amplitude_var, shift_var)
            kernel = GeometricAnisotropy(base_kernel, lambda1_var, lambda2_var, angle_var)
            trainable_variables = [v.variables[0] for v in [
                lambda1_var,
                lambda2_var,
                angle_var,
                observation_noise_variance_var,
                bias_amplitude_var, 
                slope_amplitude_var, 
                shift_var
                ]]
            variables = [lambda1_var, lambda2_var, angle_var, bias_amplitude_var, slope_amplitude_var, shift_var, observation_noise_variance_var]
        if tfk_kernel == "exp_quad":
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

    if mode == "isotropy":
        if tfk_kernel == "linear":
            kernel = tfk.Linear(bias_amplitude_var, slope_amplitude_var, shift_var)
            trainable_variables = [v.variables[0] for v in [
                bias_amplitude_var, 
                slope_amplitude_var, 
                shift_var,
                observation_noise_variance_var
                ]]
            variables = [bias_amplitude_var, slope_amplitude_var, shift_var, observation_noise_variance_var]

        if tfk_kernel == "exp_quad":
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

    return kernel, mean_fn, trainable_variables, observation_noise_variance_var, variables

# Specify kernel type
kernel, mean_fn, trainable_variables, observation_noise_variance_var, variables = kernel_func("anisotropy", "matern", vals_for_mean=y, mean_func="yes")

"""
Train model an tune hyperparameters 
"""
# Define mini-batch data iterator
batch_size = 128

batched_dataset = (
    tf.data.Dataset.from_tensor_slices(
        (X_train, y_train))
    .shuffle(buffer_size=len(X_train))
    .repeat(count=None)
    .batch(batch_size)
)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# Use tf.function for more efficient function evaluation
@tf.function(autograph=False, experimental_compile=False)
def gp_loss_fn(index_points, observations, i=0):
    """Gaussian process negative-log-likelihood loss function."""
    gp = tfd.GaussianProcess(
        kernel=kernel,
        index_points=index_points,
        observation_noise_variance=observation_noise_variance_var, 
        mean_fn=mean_fn
    )
    if i == 0:
        with tf.GradientTape() as tape:
            loss = -gp.log_prob(observations)
        grads = tape.gradient(loss, gp.trainable_variables)
        optimizer.apply_gradients(zip(grads, gp.trainable_variables))
    if i == 1:
        loss = -gp.log_prob(observations)
    return loss

# Training loop
batch_nlls = []  # Batch NLL for plotting
full_nll = []  # Full data NLL for plotting
nb_iterations = 5000
for i, (index_points_batch, observations_batch) in tqdm(
        enumerate(islice(batched_dataset, nb_iterations)), total=nb_iterations):
    observations_squeezed = tf.squeeze(observations_batch)
    loss = gp_loss_fn(index_points_batch, observations_batch)
    batch_nlls.append((i, loss.numpy()))
    # Evaluate 
    if i % 100 == 0:
        nll = gp_loss_fn(
            index_points=X_train,
            observations=y_train, i=1)
        full_nll.append((i, nll.numpy()))

# Show values for optimized hyperparameters
data = list([(var.variables[0].name[:-2], var.numpy()) for var in variables])
df_variables = pd.DataFrame(
    data, columns=['Hyperparameters', 'Value'])
df_variables

"""
Plot the loss function
"""
# Create traces for the batch data and full data plots
batch_trace = go.Scatter(x=[item[0] for item in batch_nlls],
                         y=[item[1] for item in batch_nlls],
                         mode='lines',
                         name='Batch data')

full_trace = go.Scatter(x=[item[0] for item in full_nll],
                        y=[item[1] for item in full_nll],
                        mode='lines',
                        name='All observed data',
                        yaxis='y2')

# Create figure layout
layout = go.Layout(title='Negative Log Marginal Likelihood (NLML) during training',
                   xaxis=dict(title='Iteration'),
                   yaxis=dict(title='NLML batch'),
                   yaxis2=dict(title='NLML all', overlaying='y', side='right'))

# Add traces to figure and plot
fig = go.Figure(data=[batch_trace, full_trace], layout=layout)
fig.show()

# %%
"""
Posterior predictions 
"""

gp_posterior_predict = tfd.GaussianProcessRegressionModel(
    kernel=kernel,
    index_points=X_test,
    observation_index_points=X_train,
    observations=y_train,
    observation_noise_variance=observation_noise_variance_var,
    predictive_noise_variance=0.)

# Posterior mean and standard deviation
posterior_mean_predict = gp_posterior_predict.mean()
posterior_std_predict = gp_posterior_predict.stddev()

"""Evaluate model performance"""
mean_preds = posterior_mean_predict.numpy()
rmse = mean_squared_error(y_test, mean_preds, squared=False)
r_two = r2_score(y_test, mean_preds)
crps = ps.crps_gaussian(y_test, mean_preds, posterior_std_predict).mean()
print(rmse, r_two, crps)


"""Plot uncertainty and prediction maps"""
# Create meshgrid
x, y = np.meshgrid(np.linspace(-3, 2, 100), np.linspace(-3, 2, 100))
X = np.column_stack((x.ravel(), y.ravel()))

gp_posterior_predict = tfd.GaussianProcessRegressionModel(
    kernel=kernel,
    index_points=X,
    observation_index_points=X_train,
    observations=y_train,
    observation_noise_variance=observation_noise_variance_var,
    predictive_noise_variance=0.)

# Posterior mean and standard deviation
posterior_mean_predict = gp_posterior_predict.mean()
posterior_std_predict = gp_posterior_predict.stddev()

print(min(posterior_mean_predict), max(posterior_mean_predict))
print(min(posterior_std_predict), max(posterior_std_predict))


# Scale back the predicted mean and standard deviation
y_pred = scalery.inverse_transform(np.array(posterior_mean_predict).reshape(-1, 1))
y_std = scalery.inverse_transform(np.array(posterior_std_predict).reshape(-1, 1))

# Reshape the predictions to match the meshgrid shape
z_pred = y_pred.ravel()
z_std = y_std.ravel()
# Axis limits for the plots
print(np.min(z_pred), np.max(z_pred))
print(np.min(z_std), np.max(z_std))

fig = make_subplots(rows=1, cols=2, subplot_titles=("Interpolation", "Uncertainty estimation"), shared_yaxes=True, horizontal_spacing=0.2)

# Contour fringes of the kriging process:
fig.add_trace(go.Contour(x=x.flatten(), y=y.flatten(), z=z_pred, colorscale='Plasma', 
                         autocontour=False, contours=dict(start=-20, end=57), 
                         opacity=0.5, showscale=True, colorbar=dict(x=0.42, title='Mean prediction')), row=1, col=1)

# We add the location of points of our dataset
fig.add_trace(go.Scatter(x=X_train[:,0], y=X_train[:,1], mode='markers', marker_line_color="black", marker_line_width=0.5, marker=dict(color=y_train, colorscale='Plasma', size=5), name="Data"), row=1, col=1)

# Contour fringes of the kriging error:
fig.add_trace(go.Contour(x=x.flatten(), y=y.flatten(), z=posterior_std_predict, colorscale='Plasma', 
                         autocontour=False, contours=dict(start=0, end=1), 
                         opacity=0.5, showscale=True, colorbar=dict(title='Error estimation')), row=1, col=2)

# We add the location of points of our dataset
fig.add_trace(go.Scatter(x=X_train[:,0], y=X_train[:,1], mode='markers', marker_line_color="black", marker_line_width=0.5, marker=dict(color=y_train, colorscale='Plasma', size=5), name="Data"), row=1, col=2)

# Update layout
fig.update_layout(
    title = "Anisotropic kernel",
    width=1200, 
    height=500, 
    showlegend=False, 
    xaxis=dict(title="X"), 
    yaxis=dict(title="Y")
    )

fig.show()

