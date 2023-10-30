import numpy as np
import pickle
from scipy.optimize import minimize

with open('./vars/x_init22.pkl', 'rb') as f:
    x_inits = pickle.load(f)

with open('./vars/y_init22.pkl', 'rb') as f:
    y_inits = pickle.load(f)

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern

kernel = 1.0 * Matern(length_scale=[10.4,9.65,15.15], nu=2.5)
gauss_pr = GaussianProcessRegressor(kernel)
mus=[]
mu=0
sigmas=[]
sigma=0
ucbs=[]


def _get_neg_upper_confidence_bound(x_new, gauss_pr):
    # Using estimate from Gaussian surrogate instead of actual function for
    # a new trial data point to avoid cost

    mean_y_new, sigma_y_new = gauss_pr.predict(np.array([x_new]), return_std=True)

    kappa = 3.1

    neg_ucb = -1 * mean_y_new - kappa * sigma_y_new

    global mu
    mu= mean_y_new[0]

    global sigma
    sigma= sigma_y_new[0]

    return neg_ucb


def _acquisition_function(x, gauss_pr):
    return _get_neg_upper_confidence_bound(x, gauss_pr)

for i in range(4,22):
    gauss_pr.fit(x_inits[0:i+1],y_inits[0:i+1])
    response = minimize(fun=_acquisition_function, x0=x_inits[i+1], args=(gauss_pr,), method='BFGS')

    ucb= response.fun

    mus.append(mu)
    sigmas.append(sigma)

    ucbs.append(ucb)

for i in range(4,22):

    print(x_inits[i+1],y_inits[i+1],mus[i-4],sigmas[i-4],ucbs[i-4])

X=np.array([\
    [100**2,120**2,410**2,370**2,100*120,120*410,410*370,100*370,120*370,100*410,1],\
    [110**2,120**2,408**2,362**2,110*120,120*408,408*362,110*362,120*362,110*408,1],\
    [120**2,120**2,420**2,340**2,120*120,120*420,420*340,120*340,120*340,120*420,1],\
    [128**2,121**2,433**2,318**2,128*121,121*433,433*318,128*318,121*318,128*433,1],\
    [142**2,117**2,432**2,309**2,142*117,117*432,432*309,142*309,117*309,142*432,1],\
    [131**2,101**2,452**2,316**2,131*101,101*452,452*316,131*316,101*316,131*452,1],\
    [103**2,136**2,460**2,301**2,103*136,136*460,460*301,103*301,136*301,103*460,1],\
    [119**2,130**2,396**2,355**2,119*130,130*396,396*355,119*355,130*355,115*396,1],\
    [101**2,152**2,382**2,365**2,101*152,152*382,382*365,101*365,152*365,101*382,1],\
    [100**2,157**2,413**2,330**2,100*157,157*413,413*330,100*330,157*330,100*413,1],\
    [101**2,137**2,432**2,330**2,101*137,137*432,432*330,101*330,137*330,101*432,1]])

y=np.array([185.32,314.58,701.1,202.06,161.14,465,459,417,445,556,672])

W,b = np.linalg.lstsq(X,y,rcond=None)[0][:10],np.linalg.lstsq(X,y,rcond=None)[0][10]

print(W)
print(b)