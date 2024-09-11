import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.decomposition import PCA
from functions import Database, preprocess_data, prep_data, BunDLeNet, train_model, plotting_neuronal_behavioural, plot_latent_timeseries, plot_phase_space, rotating_plot
from sklearn.preprocessing import LabelEncoder

# load data

# for final layer, simply use df[768:]

X = pd.read_csv("data_2/racetrack-loop-highway/sac/test_high_neurons.csv").to_numpy()
B = pd.read_csv("data_2/racetrack-loop-highway/sac/test_high_actions.csv").to_numpy()
O = pd.read_csv("data_2/racetrack-loop-highway/sac/test_high_obs.csv").to_numpy()

# last 256 neurons (final layer in policy function)
X = X[:, -256:]

# access only the assigned labels
B = B[:, 2]

le = LabelEncoder()
le.fit(B)
B = le.transform(B)

# get the states and plot
state_names = ['left', 'right', 'straight']
plotting_neuronal_behavioural(X, B, state_names=state_names)

# preprocess data
fps = 1
time, X = preprocess_data(X, fps)
X_, B_ = prep_data(X, B, win=2)

### Deploy BunDLe Net
model = BunDLeNet(latent_dim=3)
model.build(input_shape=X_.shape)
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)

loss_array = train_model(
	X_,
	B_,
	model,
	optimizer,
	gamma=0.9,
	n_epochs=2000,
	pca_init=False,
	best_of_5_init=False
)

# plot losses
for i, label in  enumerate(["$\mathcal{L}_{{Markov}}$", "$\mathcal{L}_{{Behavior}}$","Total loss $\mathcal{L}$" ]):
	plt.semilogy(loss_array[:,i], label=label)
plt.legend()
plt.show()

### Projecting into latent space
Y0_ = model.tau(X_[:,0]).numpy()

# plot latent time series (colour coded by actions)
# plot_latent_timeseries(Y0_, B_, state_names, figsize=(8,6))
plot_latent_timeseries(Y0_, B_, state_names)

# plot phase space in 2D
plot_phase_space(Y0_, B_, state_names = state_names)

# plot phase space in 3D (rotating plot)
rotating_plot(Y0_, B_, state_names=state_names, filename='results/first_plot.gif' ,legend=False)

### Performing PCA on the latent dimension (to check if there are redundant or correlated components)
pca = PCA()
Y_pca = pca.fit_transform(Y0_)
plot_latent_timeseries(Y_pca, B_, state_names)

### Recurrence plot analysis of embedding
pd_Y = np.linalg.norm(Y0_[:, np.newaxis] - Y0_, axis=-1) < 0.8
plt.matshow(pd_Y, cmap='Greys')
plot_latent_timeseries(Y0_, B_, state_names)
plt.show()
