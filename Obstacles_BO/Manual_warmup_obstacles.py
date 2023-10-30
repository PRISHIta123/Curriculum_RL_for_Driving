import pickle
import numpy as np
import pandas as pd

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern

kernel = 1.0 * Matern(length_scale=[10.4,9.65,15.15], nu=2.5)
gauss_pr = GaussianProcessRegressor(kernel)

x_init=[]
y_init=[]

x_init.append(np.array([100,220,630]))
x_init.append(np.array([110,230,638]))
x_init.append(np.array([120,240,660]))
x_init.append(np.array([128,249,682]))
x_init.append(np.array([142,259,691]))

y_init.append(185.32)
y_init.append(314.58)
y_init.append(701.1)
y_init.append(202.06)
y_init.append(161.14)

x_init = np.array(x_init)
y_init = np.array(y_init)

y_max_ind = np.argmax(y_init)
y_max = y_init[y_max_ind]
optimal_x = x_init[y_max_ind]
optimal_acq = None
distances_ = []
best_samples_ = pd.DataFrame(columns=['x', 'y', 'acq'])

# Pickle x_init, y_init, gauss_pr and everything above

with open('./vars/x_init' + str(4) + '.pkl', 'wb') as f:
    pickle.dump(x_init, f)

with open('./vars/y_init' + str(4) + '.pkl', 'wb') as f:
    pickle.dump(y_init, f)

with open('./vars/gauss_pr' + str(4) + '.pkl', 'wb') as f:
    pickle.dump(gauss_pr, f)

with open('./vars/y_max_ind' + str(4) + '.pkl', 'wb') as f:
    pickle.dump(y_max_ind, f)

with open('./vars/y_max' + str(4) + '.pkl', 'wb') as f:
    pickle.dump(y_max, f)

with open('./vars/optimal_x' + str(4) + '.pkl', 'wb') as f:
    pickle.dump(optimal_x, f)

with open('./vars/optimal_acq' + str(4) + '.pkl', 'wb') as f:
    pickle.dump(optimal_acq, f)

with open('./vars/distances_' + str(4) + '.pkl', 'wb') as f:
    pickle.dump(distances_, f)

with open('./vars/best_samples_' + str(4) + '.pkl', 'wb') as f:
    pickle.dump(best_samples_, f)





