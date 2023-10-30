import numpy as np
import pickle
from scipy.optimize import minimize

with open('./vars/x_init18.pkl', 'rb') as f:
    x_inits = pickle.load(f)

with open('./vars/y_init18.pkl', 'rb') as f:
    y_inits = pickle.load(f)

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern

kernel = 1.0 * Matern(length_scale=[19.9,26.5,21.15], nu=2.5)
gauss_pr = GaussianProcessRegressor(kernel)
mus=[]
sigmas=[]
mu=0
ucbs=[]


def _get_neg_upper_confidence_bound(x_new, gauss_pr):
    # Using estimate from Gaussian surrogate instead of actual function for
    # a new trial data point to avoid cost

    mean_y_new, sigma_y_new = gauss_pr.predict(np.array([x_new]), return_std=True)

    kappa = 1.75

    neg_ucb = -1 * mean_y_new - kappa * sigma_y_new

    global mu
    mu= mean_y_new[0]

    global sigma
    sigma= sigma_y_new[0]

    return neg_ucb


def _acquisition_function(x, gauss_pr):
    return _get_neg_upper_confidence_bound(x, gauss_pr)

for i in range(4,18):
    gauss_pr.fit(x_inits[0:i+1],y_inits[0:i+1])
    response = minimize(fun=_acquisition_function, x0=x_inits[i+1], args=(gauss_pr,), method='BFGS')

    ucb= response.fun

    mus.append(mu)
    sigmas.append(sigma)

    ucbs.append(ucb)

for i in range(4,18):

    print(y_inits[i+1],mus[i-4],sigmas[i-4])

X=np.array([\
    [158**2,176**2,401**2,265**2,158*176,176*401,401*265,158*265,176*265,158*401,1],\
    [178**2,187**2,383**2,252**2,178*187,187*383,383*252,178*252,187*252,178*383,1],\
    [198**2,198**2,379**2,225**2,198*198,198*379,379*225,198*225,198*225,198*379,1],\
    [218**2,209**2,368**2,205**2,218*209,209*368,368*205,218*205,209*205,218*368,1],\
    [238**2,204**2,368**2,190**2,238*204,204*368,368*190,238*190,204*190,238*368,1],\
    [157**2,242**2,350**2,251**2,157*242,242*350,350*251,157*251,242*251,157*350,1],\
    [174**2,263**2,326**2,237**2,174*263,263*326,326*237,174*237,263*237,174*326,1],\
    [158**2,251**2,409**2,182**2,158*251,251*409,409*182,158*182,251*182,158*409,1],\
    [212**2,187**2,379**2,222**2,212*187,187*379,379*222,212*222,187*222,212*379,1],\
    [161**2,257**2,319**2,263**2,161*257,257*319,319*263,161*263,257*263,161*319,1],\
    [184**2,231**2,386**2,199**2,184*231,231*386,386*199,184*199,231*199,184*386,1]\
    ])

y=np.array([290.4,543.55,756,673.14,215.24,703,589,669,705,689,686])

W,b = np.linalg.lstsq(X,y,rcond=None)[0][:10],np.linalg.lstsq(X,y,rcond=None)[0][10]

print(W)
print(b)
