import numpy as np
import pickle
from scipy.optimize import minimize

with open('./vars/x_init18.pkl', 'rb') as f:
    x_inits = pickle.load(f)

with open('./vars/y_init18.pkl', 'rb') as f:
    y_inits = pickle.load(f)

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern

kernel = 1.0 * Matern(length_scale=[19.9,23.65,21.4], nu=2.5)
gauss_pr = GaussianProcessRegressor(kernel)
mus=[]
mu=0
ucbs=[]


def _get_neg_upper_confidence_bound(x_new, gauss_pr):
    # Using estimate from Gaussian surrogate instead of actual function for
    # a new trial data point to avoid cost

    mean_y_new, sigma_y_new = gauss_pr.predict(np.array([x_new]), return_std=True)

    kappa = 2

    neg_ucb = -1 * mean_y_new - kappa * sigma_y_new

    global mu
    mu= mean_y_new[0]

    return neg_ucb


def _acquisition_function(x, gauss_pr):
    return _get_neg_upper_confidence_bound(x, gauss_pr)

for i in range(4,18):
    gauss_pr.fit(x_inits[0:i+1],y_inits[0:i+1])
    response = minimize(fun=_acquisition_function, x0=x_inits[i+1], args=(gauss_pr,), method='BFGS')

    ucb= response.fun

    mus.append(mu)

    ucbs.append(ucb)

for i in range(4,18):

    print(y_inits[i+1],mus[i-4])

X=np.array([\
    [150**2,215**2,305**2,330**2,150*215,215*305,305*330,150*330,215*330,150*305,1],\
    [170**2,210**2,305**2,315**2,170*210,210*305,305*315,170*315,210*315,170*305,1],\
    [190**2,222**2,284**2,304**2,190*222,222*284,284*304,190*304,222*304,190*284,1],\
    [210**2,225**2,295**2,270**2,210*225,225*295,295*270,210*270,225*270,210*295,1],\
    [230**2,230**2,296**2,244**2,230*230,230*296,296*244,130*244,230*244,230*296,1],\
    [177**2,216**2,281**2,326**2,177*216,216*281,281*326,177*326,216*326,177*281,1],\
    [156**2,303**2,227**2,314**2,156*303,303*227,227*314,156*314,303*314,156*227,1],\
    [165**2,251**2,288**2,296**2,165*251,251*288,288*296,165*296,251*296,165*288,1],\
    [248**2,182**2,264**2,306**2,248*182,182*264,264*306,248*306,182*306,248*264,1],\
    [239**2,203**2,269**2,289**2,239*203,203*269,269*289,239*289,203*289,239*269,1],\
    [238**2,148**2,315**2,298**2,238*148,148*315,315*298,238*298,148*298,238*315,1]\
    ])

y=np.array([514,673.7,788.8,749.62,697.5,682,675,556,664,805,837])

W,b = np.linalg.lstsq(X,y,rcond=None)[0][:10],np.linalg.lstsq(X,y,rcond=None)[0][10]

print(W)
print(b)

