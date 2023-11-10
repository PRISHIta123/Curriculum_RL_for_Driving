import pickle
import numpy as np
import pandas as pd

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern

kernel = 1.0 * Matern(length_scale=[19.9,26.5,21.15], nu=2.5)
gauss_pr = GaussianProcessRegressor(kernel)

x_init=[]
y_init=[]

x_init.append(np.array([158,334,735]))
x_init.append(np.array([178,365,748]))
x_init.append(np.array([198,396,775]))
x_init.append(np.array([218,427,795]))
x_init.append(np.array([238,442,820]))

y_init.append(290.40)
y_init.append(543.55)
y_init.append(756)
y_init.append(673.14)
y_init.append(215.24)

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





