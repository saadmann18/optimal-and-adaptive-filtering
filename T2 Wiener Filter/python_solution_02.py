#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, linalg
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rcParams


# In[2]:


rcParams.update({'legend.fontsize': 'x-large',
                 'figure.figsize': (12, 6),
                 'axes.labelsize': 'xx-large',
                 'axes.titlesize':'x-large',
                 'xtick.labelsize':'x-large',
                 'ytick.labelsize':'x-large'})


# # P2 - Channel Equalization

# This exercise aims at implementing a Wiener filter (with a varying number of coefficients) to equalize 
#the transmission channel.

# In[3]:

beta = np.sqrt(0.27)

# In[4]:
# ## a)
# 
# Implement the channel (neglect the noise) by defining a function `channel(input_stream, coefficients)` 
# which describes the second order AR process.
# The internal states of the memory cells can be initialized with zeros.

def channel(input_stream, coefficients):
    """
    :param input_stream: Array of scaled BPSK signal beate * d(n).
    :param coefficients: Coefficients of the transfer function of the channel.
    """
    assert len(coefficients)==3
    alpha=[0,0] #channel has 2 coefficients
    for input_ in input_stream:
        alpha+=[coefficients[0]*input_ + coefficients[1]*alpha[-1] + coefficients[2]*alpha[-2]]
    return np.asarray(alpha[2:])    
    #IIR filter, that's why AR process is applied
    #the mechanism is similar to convolution


# In[5]:
# Test your channel function a little bit:

expected_output = np.asarray([1., 3., 7., 14.])
output = channel([1, 2, 3, 4], [1, 1, 1])
np.testing.assert_equal(output, expected_output)


# ## b)
# 
# Calculate the autocorrelation matrix $\mathbf R_{\alpha}$ and the cross correlation vector $\mathbf p_{\mathbf xd}$ for a Wiener filter with 3 coefficients.
# The input sequence is of variance $\sigma_d^2 = 1$.

# Hint: Check [`scipy.linalg.toeplitz`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.toeplitz.html).

# In[6]:


def get_R_alpha(num_coefficients, beta=np.sqrt(.27), channel_coefficients=[1, .1, .8], noise_variance=1.):
    assert len(channel_coefficients) == 3
    a = np.asarray([[1, -channel_coefficients[1], -channel_coefficients[2]],
                   [-channel_coefficients[1], 1-channel_coefficients[2], 0],
                   [-channel_coefficients[2], -channel_coefficients[1], 1]])
    b = np.zeros((3,1))
    b[0,0] = beta ** 2 * noise_variance    #according to filter structure, there are two delay blocks
    if num_coefficients < 3:               #asked for 3 or 2 coeffs
        akf_alpha = np.zeros(3)
    else:
        akf_alpha = np.zeros(num_coefficients)
    akf_alpha[:3] = np.linalg.solve(a,b).squeeze(-1)
    for l in range(3, num_coefficients):
        akf_alpha[l] = channel_coefficients[1]*akf_alpha[l-1]+channel_coefficients[2]*akf_alpha[l-2]
    return linalg.toeplitz(akf_alpha[:num_coefficients], akf_alpha[:num_coefficients])

def get_crosscorrelation_vector(num_coefficients, beta=np.sqrt(.27), channel_coefficients=[1, .1, .8], noise_variance=1.):
    if num_coefficients < 3:
        len_r = 3
    else:
        len_r = num_coefficients
    r_alpha = get_R_alpha(len_r, beta, channel_coefficients, noise_variance)[0]
    xcorr_vector = np.zeros(num_coefficients)
    for l in range(num_coefficients):
        xcorr_vector[l] = (r_alpha[l]
                          -channel_coefficients[1] * r_alpha[abs(l-1)]
                          -channel_coefficients[2] * r_alpha[abs(l-2)]) / beta
    return xcorr_vector


# In[7]:


get_R_alpha(3)


# In[8]:


get_crosscorrelation_vector(3)


# ## c)
# 
# Solve the Wiener-Hopf equation to estimate $d(n)$ with a Wiener filter having 3 coefficients.
# Consider the noise-free and the noisy case ($\sigma_v^2 = 0.1$).

# In[9]:


def get_R_v(num_coefficients, noise_variance=0.1):
    return noise_variance * np.eye(num_coefficients)
def get_wiener_filter(num_coefficients, noise_variance=0.1):
    x_corr_vector = get_crosscorrelation_vector(num_coefficients)
    corr_mat = get_R_alpha(num_coefficients)
    if noise_variance > 0:
        corr_mat = corr_mat + get_R_v(num_coefficients, noise_variance)
    return np.linalg.solve(corr_mat, x_corr_vector)        #calculating w = R^-1*p or wR = p


# In[10]:


get_wiener_filter(3, noise_variance=0)


# In[11]:


get_wiener_filter(3, noise_variance=.1)


# ## d)
# 
# Implement the application of the Wiener filter to the input sequence $d(n)$.

# Hint: Check [`numpy.convolve`](https://numpy.org/doc/stable/reference/generated/numpy.convolve.html).

# In[27]:


def filter_sequence(x, wf_coefficients):
    return np.convolve(x, wf_coefficients)[:-(len(wf_coefficients)-1)]


# In[38]:


d = np.asarray([1, -1, -1, 1, -1])
x = channel(beta * d, [1, .1, .8])
wf = get_wiener_filter(3, noise_variance=0.0)
d_hat = filter_sequence(x, wf)
print(f'Send: {d}, Recieved: {x}')
print(f'After channel equalization: {d_hat}')


# In[39]:


print(wf)
print(x)


# ## e)
# 
# Apply the Wiener filter to a white input sequence of $N=200$ BPSK-symbols transmitted over
# the channel ($d(n) \in\{+1,-1\}$). Again, consider $v(n)=0$ and $v(n)\neq0$.
# Compute the mean squared error for a wiener filter with 2, 3, and 4 coefficients, using $200$ 
# observations for $v(n)=0$ and $v(n)\neq0$. Visualize  the error function for a Wiener filter with 2 coefficients.

# In[14]:


def apply_wiener_filter(N, noise_variance, num_coefficients, var_d=1):
    def mse(d, d_hat):
        return np.mean((d - d_hat)**2)
          
    channel_coefficients = [1, 0.1, 0.8]     #unique only for this channel diagram
    
    wf = get_wiener_filter(num_coefficients, noise_variance)      #coeffs found for this channel diagram
    
    d = np.random.choice((-1, 1), N)         #BPSK input sequence d(n) to the channel
    
    alpha = channel(beta * d, channel_coefficients)               #2nd order AR process in channel
    v = np.random.normal(scale=np.sqrt(noise_variance), size=alpha.shape)
    x = alpha + v
    d_hat = filter_sequence(x, wf)
    
    print('Optimal coefficients: ', wf)
    
    # Analytical MSE
    x_corr_vector = get_crosscorrelation_vector(num_coefficients)
    R_x = get_R_alpha(num_coefficients)
    if noise_variance > 0:
        R_x += get_R_v(num_coefficients, noise_variance)
    j_mse = np.squeeze(var_d - x_corr_vector.T @ np.linalg.inv(R_x) @ x_corr_vector)
    print('Analytic MSE-Error: ', j_mse)
    
    # Estimated MSE
    j_mse_hat = mse(d, d_hat)
    print('Estimated MSE-Error: ' ,j_mse_hat)


# $v(n)=0$:

# In[15]:


apply_wiener_filter(N=200, noise_variance=0, num_coefficients=2)


# In[16]:


apply_wiener_filter(N=200, noise_variance=0, num_coefficients=3)


# In[17]:


apply_wiener_filter(N=200, noise_variance=0, num_coefficients=4)


# $v(n) \neq 0$:

# In[18]:


apply_wiener_filter(N=200, noise_variance=.1, num_coefficients=2)


# In[19]:


apply_wiener_filter(N=200, noise_variance=.1, num_coefficients=3)


# In[20]:


apply_wiener_filter(N=200, noise_variance=.1, num_coefficients=4)


# ### Error plane (for 2 coefficients):

# In[47]:


p_xd = get_crosscorrelation_vector(2)
r_x = get_R_alpha(2)[0]
print(p_xd)
print(r_x)
x = np.arange(-2, 2, 0.1)
y = np.arange(-2, 2, 0.1)
xx, yy = np.meshgrid(x, y, sparse=True)

j_mse = 1 - 2 * (xx * p_xd[0] + yy * p_xd[1]) + r_x[0] * (xx ** 2 + yy ** 2) + 2 * r_x[1] * xx * yy
#j_mse is calculated from the last page of theoratical exercise
plt.contourf(x, y, j_mse, cmap=cm.viridis)
plt.colorbar()
plt.xlabel('$w_o(0)$')
plt.ylabel('$w_o(1)$')
plt.show()


# In[22]:


fig = plt.figure(figsize=(20, 10))
# ax = fig.gca(projection='3d')
ax = fig.add_subplot(projection='3d')
ax.plot_surface(
    xx, yy, j_mse, cmap=cm.viridis, linewidth=1., rstride=4, cstride=4, edgecolor='w', antialiased=True
)
ax.set_xlabel('\n $w_o(0)$')
ax.set_ylabel('\n $w_o(1)$')
ax.set_zlabel('E$[|e(n)|^2]$')
m = cm.ScalarMappable(cmap=cm.viridis)
m.set_array(j_mse)
plt.colorbar(m)
plt.show()


# In[ ]:




