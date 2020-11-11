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

# In[3]:

beta = np.sqrt(0.27)

# In[6]:
# channel function analysis
input_stream = [1, 2, 3, 4]
coefficients = [1, 1, 1]
"""
:param input_stream: Array of scaled BPSK signal beate * d(n).
:param coefficients: Coefficients of the transfer function of the channel.
"""
alpha=[0,0]
for input_ in input_stream:
    #alpha+=[coefficients[0]*input_ + coefficients[1]*alpha[-1] + coefficients[2]*alpha[-2]]
    alpha = alpha + [coefficients[0]*input_ + coefficients[1]*alpha[-1] + coefficients[2]*alpha[-2]]
    
output =  np.asarray(alpha[2:])



# In[8]:
# # get_R_alpha function analysis : Auto correlation matrix
num_coefficients = 3 
beta = np.sqrt(0.27)
channel_coefficients=[1, .1, .8]
noise_variance=1.
assert len(channel_coefficients) == 3
a = np.asarray([[1, -channel_coefficients[1], -channel_coefficients[2]],
                [-channel_coefficients[1], 1-channel_coefficients[2], 0],
                [-channel_coefficients[2], -channel_coefficients[1], 1]]) #New way to get similar result?
b = np.zeros((3,1))
b[0,0] = beta ** 2 * noise_variance    
if num_coefficients < 3:               #asked for 3 coeffs 
    akf_alpha = np.zeros(3)
else:
    akf_alpha = np.zeros(num_coefficients)
akf_alpha[:3] = np.linalg.solve(a,b).squeeze(-1)

#for more than 3 coefficients:
for l in range(3, num_coefficients):
    akf_alpha[l] = channel_coefficients[1]*akf_alpha[l-1]+channel_coefficients[2]*akf_alpha[l-2]
R_alpha =  linalg.toeplitz(akf_alpha[:num_coefficients], akf_alpha[:num_coefficients])

# In[9]:
# # get_cross_correlation_vector function analysis, results from theoratical part
if num_coefficients < 3:
    len_r = 3
else:
    len_r = num_coefficients
r_alpha = R_alpha[0]
xcorr_vector = np.zeros(num_coefficients)
for l in range(num_coefficients):
    xcorr_vector[l] = (r_alpha[l]
                      -channel_coefficients[1] * r_alpha[abs(l-1)]
                      -channel_coefficients[2] * r_alpha[abs(l-2)]) / beta

# #xcorr_vector is p_xd from theory which equals to sqrt(0.27) =  0.519615

# In[11]:
# noise_variance=0.0 #0.0
# R_v = noise_variance * np.eye(num_coefficients)
# x_corr_vector = xcorr_vector
# corr_mat = R_alpha

# if noise_variance > 0:
#         corr_mat = corr_mat + R_v

# wiener_filter = np.linalg.solve(corr_mat, x_corr_vector)

# #calculating w = R^-1*p or wR = p

# # In[ ]:
# #From here, jupyter notebook solution
# def filter_sequence(x, wf_coefficients):
#     return np.convolve(x, wf_coefficients)[:-(len(wf_coefficients) -1 )]

# # In[ ]:


# d = np.asarray([1, -1, -1, 1, -1])
# x = channel(beta * d, [1, .1, .8])
# wf = get_wiener_filter(3, noise_variance=0)
# d_hat = filter_sequence(x, wf)
# print(f'Send: {d}, Recieved: {x}')
# print(f'After channel equalization: {d_hat}')


# # ## e)
# # 
# # Apply the Wiener filter to a white input sequence of $N=200$ BPSK-symbols transmitted over
# # the channel ($d(n) \in\{+1,-1\}$). Again, consider $v(n)=0$ and $v(n)\neq0$.
# # Compute the mean squared error for a wiener filter with 2, 3, and 4 coefficients, using $200$ 
# # observations for $v(n)=0$ and $v(n)\neq0$. Visualize  the error function for a Wiener filter with 2 coefficients.

# # In[ ]:


# def apply_wiener_filter(N, noise_variance, num_coefficients, var_d=1):
#     def mse(d, d_hat):
#         raise NotImplementedError
        
#     channel_coefficients = [1, 0.1, 0.8]
    
#     wf = get_wiener_filter(num_coefficients, noise_variance)
    
#     d = np.random.choice((-1, 1), N)
    
#     alpha = channel(beta * d, channel_coefficients)
#     v = np.random.normal(scale=np.sqrt(noise_variance), size=alpha.shape)
#     x = alpha + v
#     d_hat = filter_sequence(x, wf)
    
#     print('Optimal coefficients: ', wf)
    
#     # Analytical MSE
#     j_mse = ???
#     print('Analytic MSE-Error: ', j_mse)
    
#     # Estimated MSE
#     j_mse_hat = mse(d, d_hat)
#     print('Estimated MSE-Error: ' ,j_mse_hat)


# # $v(n)=0$:

# # In[ ]:


# apply_wiener_filter(N=200, noise_variance=0, num_coefficients=2)


# # In[ ]:


# apply_wiener_filter(N=200, noise_variance=0, num_coefficients=3)


# # In[ ]:


# apply_wiener_filter(N=200, noise_variance=0, num_coefficients=4)


# # $v(n) \neq 0$:

# # In[ ]:


# apply_wiener_filter(N=200, noise_variance=.1, num_coefficients=2)


# # In[ ]:


# apply_wiener_filter(N=200, noise_variance=.1, num_coefficients=3)


# # In[ ]:


# apply_wiener_filter(N=200, noise_variance=.1, num_coefficients=4)


# # Error plane (for 2 coefficients):

# # In[ ]:


# p_xd = get_crosscorrelation_vector(2)
# r_x = get_R_alpha(2)[0]

# x = np.arange(-2, 2, 0.1)
# y = np.arange(-2, 2, 0.1)
# xx, yy = np.meshgrid(x, y, sparse=True)

# j_mse = 1 - 2 * (xx * p_xd[0] + yy * p_xd[1]) + r_x[0] * (xx ** 2 + yy ** 2) + 2 * r_x[1] * xx * yy
# plt.contourf(x, y, j_mse, cmap=cm.viridis)
# plt.colorbar()
# plt.xlabel('$w_o(0)$')
# plt.ylabel('$w_o(1)$')
# plt.show()


# # In[ ]:


# fig = plt.figure(figsize=(20, 10))
# ax = fig.gca(projection='3d')
# ax.plot_surface(
#     xx, yy, j_mse, cmap=cm.viridis, linewidth=1., rstride=4, cstride=4, edgecolor='w', antialiased=True
# )
# ax.set_xlabel('\n $w_o(0)$')
# ax.set_ylabel('\n $w_o(1)$')
# ax.set_zlabel('E$[|e(n)|^2]$')
# m = cm.ScalarMappable(cmap=cm.viridis)
# m.set_array(j_mse)
# plt.colorbar(m)
# plt.show()


# In[ ]:




