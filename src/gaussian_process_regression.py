# %%
"""
Import libraries
"""
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

from kernel_setup import kernel_setup

# Metrics
import properscoring as ps
from sklearn.metrics import r2_score, mean_squared_error

# %%
"""
Load data 
"""
filename = open("C:/Users/Maris/Documents/GitHub/master/data/Grunnboringer_OsloOst.csv")
data = pd.read_csv(filename, header=0)

"""
Data preprocessing 
"""
# Check and remove rows containing NaN values
data.replace('<Null>', np.nan, inplace = True)
df = data.dropna(subset=['DTB'])
# Cast DTB column to float
df['DTB'] = df['DTB'].astype(np.float64)

# filter rows with negative values in column 'A'
#df = df[df['DTB'] >= 0]

# Split data to x and y
y = df.DTB.values.reshape(-1, 1)
x = np.array([df.X, df.Y]).T 

# Split dataset into test and training sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

# Scale data with StandardScaler
scalerX = StandardScaler().fit(X_train)
scalery = StandardScaler().fit(y_train)

X_train_sc = scalerX.transform(X_train)
y_train_sc = scalery.transform(y_train)
X_test_sc = scalerX.transform(X_test)
y_test_sc = scalery.transform(y_test)

# Convert y_train and y_test to 1-dimensional arrays
y_train_sc = y_train_sc.ravel()
y_test_sc = y_test_sc.ravel()

# %%
"""
Train model and tune hyperparameters 
"""
# Specify kernel type
kernel, mean_fn, trainable_variables, observation_noise_variance_var, variables = kernel_setup(
    "anisotropy", "matern"
    )

# Define mini-batch data iterator
batch_size = 128
batched_dataset = (
    tf.data.Dataset.from_tensor_slices(
        (X_train_sc, y_train_sc))
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
nb_iterations = 500
for i, (index_points_batch, observations_batch) in tqdm(
        enumerate(islice(batched_dataset, nb_iterations)), total=nb_iterations):
    loss = gp_loss_fn(index_points_batch, observations_batch)
    batch_nlls.append((i, loss.numpy()))
    # Evaluate 
    if i % 100 == 0:
        nll = gp_loss_fn(
            index_points=X_train_sc,
            observations=y_train_sc, i=1)
        full_nll.append((i, nll.numpy()))

# Save values for optimized hyperparameters
data = list([(var.variables[0].name[:-2], var.numpy()) for var in variables])
df_variables = pd.DataFrame(
    data, columns=['Hyperparameters', 'Value'])

# Save loss results
pd.DataFrame(batch_nlls).to_csv("nlls.csv")

# %%
"""
Posterior predictions on test data
"""

gp_posterior_predict = tfd.GaussianProcessRegressionModel(
    kernel=kernel,
    index_points=X_test_sc,
    observation_index_points=X_train_sc,
    observations=y_train_sc,
    observation_noise_variance=observation_noise_variance_var,
    predictive_noise_variance=0.)

# Posterior mean and standard deviation
posterior_mean_predict = gp_posterior_predict.mean()
posterior_std_predict = gp_posterior_predict.stddev()


# Evaluate model performance
mean_preds = posterior_mean_predict.numpy()
rmse = mean_squared_error(y_test_sc, mean_preds, squared=False)
r_two = r2_score(y_test_sc, mean_preds)
crps = ps.crps_gaussian(y_test_sc, mean_preds, posterior_std_predict).mean()
print(rmse, r_two, crps)

# Inverse transform data to its original scale
y_pred = (scalery.inverse_transform(np.array(posterior_mean_predict).reshape(-1, 1))).ravel()
y_std = (scalery.inverse_transform(np.array(posterior_std_predict).reshape(-1, 1))).ravel()

# Save predictions
results = pd.DataFrame(data={
    'X_test': X_test[:,0], 'Y_test': X_test[:,1], 'y_pred': y_pred, 'y_std': y_std
    })
results.to_csv(path_or_buf='results.csv', index = False, header = True)

# %%
"""
Interpolation
"""
# Create meshgrid
x, y = np.meshgrid(np.linspace(-2, 2.5, 100), np.linspace(-2, 2.5, 100))
X = np.column_stack((x.ravel(), y.ravel()))

gp_posterior_predict = tfd.GaussianProcessRegressionModel(
    kernel=kernel,
    index_points=X,
    observation_index_points=X_train_sc,
    observations=y_train_sc,
    observation_noise_variance=observation_noise_variance_var,
    predictive_noise_variance=0.)

# Posterior mean and standard deviation
posterior_mean_predict = gp_posterior_predict.mean()
posterior_std_predict = gp_posterior_predict.stddev()

# Scale back the predicted mean and standard deviation
y_pred = scalery.inverse_transform(np.array(posterior_mean_predict).reshape(-1, 1))
y_std = scalery.inverse_transform(np.array(posterior_std_predict).reshape(-1, 1))
y_train = scalery.inverse_transform(np.array(y_train_sc).reshape(-1, 1))

# Reshape the predictions to match the meshgrid shape
z_pred = y_pred.ravel()
z_std = y_std.ravel()

# Scale back coordinates
X_train = scalerX.inverse_transform(X_train_sc)
X = scalerX.inverse_transform(X)

"""
Save results
"""
result = pd.DataFrame(data={'x': X[:,0], 'y': X[:,1], 'predictions': z_pred})
result.to_csv(path_or_buf='mean_pred_interpolation.csv', index = False, header = True)
error = pd.DataFrame(data={'x': X[:,0], 'y': X[:,1], 'std': z_std})
error.to_csv(path_or_buf='std_interpolation.csv', index = False, header = True)
