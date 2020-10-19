#!/usr/bin/env python
# coding: utf-8

# In[1]:
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib import rcParams

# In[2]:

rcParams.update({'legend.fontsize': 'x-large',
                 'figure.figsize': (12, 6),
                 'axes.labelsize': 'x-large',
                 'axes.titlesize':'x-large',
                 'xtick.labelsize':'x-large',
                 'ytick.labelsize':'x-large'})

# In[3]:

sigma_nu = np.sqrt(1)
theta = np.random.choice([-1, 1]) #symbol alphabet.

nu = np.random.normal(0.0, sigma_nu)
x = theta + nu

# In[6]:

def p_x_given_theta(x, theta, sigma_nu):
    """
    
    Parameters
    ----------
    x : float
        unknown received signal.
    theta : int
        symbol alphabet{-1, 1}.
    sigma_nu : float
        gaussian distributed, zero-mean, unit-variance noise.

    Returns
    -------
    float
        a pdf made with x values, location at {-1, 1} and with sigma_nu spread.

    """
    return stats.norm.pdf(x, loc=theta, scale=sigma_nu) 

def p_x(x, sigma_nu):
    """
    
    Parameters
    ----------
    x : float
        unknown received signal.
    sigma_nu : float
        gaussian distributed, zero-mean, unit-variance noise.
        
    Returns
    -------
    float
        complete pdf of signal x, derrived from joint pdf by integrating w.r.t.
        the symbol alphabet theta.

    """
    return 0.5*stats.norm.pdf(x,loc=-1,scale=sigma_nu) + 0.5*stats.norm.pdf(x,loc=1,scale=sigma_nu)

def p_theta_given_x(theta, x, sigma_nu):
    """
    
    Parameters
    ----------
    theta : int
        symbol alphabet{-1, 1}.
    x : float
        unknown received signal.
    sigma_nu : float
        gaussian distributed, zero-mean, unit-variance noise.

    Raises
    ------
    ValueError
        the value is not in the symbol alphabet.

    Returns
    -------
    float
        Posterior probability, decesion is taken based on the largest value.

    """
    a = np.exp(-x / sigma_nu**2)
    b = np.exp(x / sigma_nu**2)
    if theta == -1:
        return a / (a+b)
    elif theta == 1:
        return b / (a+b)
    else:
        raise ValueError
        
def p_error_given_x(x, sigma_nu):
    """
    

    Parameters
    ----------
    x : float
        unknown received signal.
    sigma_nu : float
        gaussian distributed, zero-mean, unit-variance noise.

    Returns
    -------
    error curve, float value
        # np.minimum-> Fig 1c: choosing bottom curve always leads to wrong value..

    """
    return np.minimum(   
        p_theta_given_x(-1, x, sigma_nu),
        p_theta_given_x(1, x, sigma_nu)
    )

# In[7]:

x = np.arange(-2.5, 2.5, 0.01)
for sigma in [np.sqrt(.1), np.sqrt(.5), np.sqrt(1.)]:
    plt.plot(x, p_x_given_theta(x, -1, sigma))
    plt.plot(x, p_x_given_theta(x, 1, sigma))
    plt.xlabel('$x$')
    plt.ylabel(r'$p(x | \theta)$')
    plt.title(r'$\sigma_\nu^2={}$'.format(np.round(sigma**2, 2)))
    plt.legend([r'$\theta = -1$', r'$\theta = +1$'], loc='upper right')
    plt.show()

# In[9]:

x = np.arange(-2.5, 2.5, 0.01)
for sigma in [np.sqrt(.1), np.sqrt(.5), np.sqrt(1.)]:
    plt.plot(x, p_x(x, sigma))
    plt.xlabel('$x$')
    plt.ylabel('$p(x)$')
    plt.title(r'$\sigma_\nu^2={}$'.format(np.round(sigma**2, 2)))
    plt.show()

# In[10]:

x = np.arange(-2.5, 2.5, 0.01)
for sigma in [np.sqrt(.1), np.sqrt(.5), np.sqrt(1.)]:
    plt.plot(x, p_theta_given_x(-1, x, sigma))
    plt.plot(x, p_theta_given_x(1, x, sigma))
    plt.xlabel('$x$')
    plt.ylabel(r'$Pr(\theta| x)$')
    plt.legend([r'$\theta = -1$', r'$\theta = +1$'], loc='upper right')
    plt.title(r'$\sigma_\nu^2={}$'.format(np.round(sigma**2, 2)))
    plt.show()


# In[11]:

x = np.arange(-2.5, 2.5, 0.01)
for sigma in [np.sqrt(.1), np.sqrt(.5), np.sqrt(1.)]:
    plt.plot(x, p_error_given_x(x, sigma))
    plt.xlabel('$x$')
    plt.ylabel('$Pr(error | x)$')
    plt.title(r'$\sigma_\nu^2={}$'.format(np.round(sigma**2, 2)))
    plt.show()

# In[12]:

def mmse_estimator(x, sigma_nu):
    return np.tanh(np.sum(x) / sigma_nu ** 2)    #this is the result of MMSE E[theta|x] 

def avg_estimator(x):
    return np.mean(x)                            #this is just the Expected value

# In[13]:

range_N = np.arange(1, 200, dtype=int)
mmse_estimate = np.zeros(shape=range_N.shape)
avg_estimate = np.zeros(shape=range_N.shape)
for index, n in enumerate(range_N):
    nu = np.random.normal(0.0, sigma_nu, size=(n,))
    x = theta + nu
    mmse_estimate[index] = mmse_estimator(x, sigma_nu)
    avg_estimate[index] = avg_estimator(x)
    
plt.plot(range_N, mmse_estimate)
plt.plot(range_N, avg_estimate)
plt.xlabel('Number of observations N')
plt.ylabel(r'$\hat{\theta}(\mathbf{x})$')
plt.legend([r'$\hat{\theta}(\mathbf{x})=\hat{\theta}^{(MMSE)}(\mathbf{x})$', r'$\hat{\theta}(\mathbf{x})=\hat{\theta}^{(AVG)}(\mathbf{x})$'], loc='upper right')
plt.show()

# In[14]:

#initialization
mse_mmse = 0.
mse_avg = 0.
mse_sgn_mmse = 0.
trials = 10000
N = 10
for _ in range(trials):
    theta = np.random.choice([-1,1])
    nu = np.random.normal(0.0, sigma_nu, size=(N,))
    x = theta + nu
    mse_mmse += (mmse_estimator(x, sigma_nu) - theta) **2
    mse_avg += (avg_estimator(x) - theta) ** 2
    mse_sgn_mmse += (np.sign(mmse_estimator(x,sigma_nu))-theta) ** 2
mse_mmse /=trials
mse_avg /=trials
mse_sgn_mmse /=trials

print('MSE of MMSE estimator: {}'.format(mse_mmse))
print('MSE of AVG estimator: {}'.format(mse_avg))
print('MSE of SGN-MMSE estimator: {}'.format(mse_sgn_mmse))
