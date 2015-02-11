# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# #Gaussian Processes Tutorial
# ####Ahsan S. Alvi, Elmarie van Heerden
# 
# Clone this tutorial with the following command: "git clone https://github.com/a5a/GP_tutorial.git" (without the quotation marks)
# 
# This tutorial makes use of the Gaussian Processes framework GPy  (https://github.com/SheffieldML/GPy). If you don't have this installed yet, follow one of the following installation instructions:
# 
# A. Github (preferred)
#   1. In a terminal, change to the directory where you want to store the library's files
#   2. Run "git clone https://github.com/SheffieldML/GPy.git" (without the quotation marks)
# 
# B. Python-pip
#   1. Run "pip install gpy" (without the quotation marks)
# 
# 
# Other requirements are:
#   1. numpy
#   2. scipy
#   3. matplotlib
# 
# 
# Acknowledgement: Parts of this tutorial are inspired by the tutorials given at the Gaussian Process Summer School 2013 (http://ml.dcs.shef.ac.uk/gpss/gpss13)

# <codecell>

"""
If this cell works, then you have all the requirements 
for this tutorial installed correctly!
"""
import numpy as np
import pylab as plt
%matplotlib inline

"""
Adding the location of the library should only be necessary 
if you haven't already added GPy to your PYTHONPATH
"""
from os.path import expanduser as eu
import sys
# Change this to your GPy installation location
GPy_path = '~/DPhil/Code/libraries/GPy'  
sys.path.append(eu(GPy_path))

import GPy

# <codecell>

#Misc settings
fs = (9, 5)  # figure size; change this if you need to
ms = 11  # marker size in plots
sd = 6  # random seed

# <markdowncell>

# ##Kernels
# The workhorse of the Gaussian Process is the kernel. This defines the relationship between points and contains any prior knowledge of the domain:
# 
# \begin{equation}
# \mathbf{f} \sim \mathcal{GP}(0, k(\mathbf{x},\mathbf{x})).
# \end{equation}
# 
# The kernels are accessible via the GPy.kern module. A large number of kernels are available out of the box and they all support adding and multiplication with other kernels. New kernels can be developed by extending the kernel class.
# 
# We will first look at one of the most popular kernels - the squared exponential kernel, also known as the exponentiated quadratic or radial basis function (RBF) kernel:
# \begin{equation}
# k(x_1, x_2) = \sigma^2 \exp \left(-\frac{\left|x_1 - x_2\right|^2}{2 l^2} \right).
# \end{equation}
# 
# See the accompanying presentation for the functional form of some popular kernels. 

# <codecell>

# Note: Supplying kernel parameters is optional.

k = GPy.kern.RBF(1, variance=1.5, lengthscale=0.4)
# k = GPy.kern.Matern32(1)
# k = GPy.kern.Matern52(1)
# k = GPy.kern.Exponential(1)
# k = GPy.kern.PeriodicExponential(1)

# We are provided a summary of the kernel by printing the variable
print k, '\n\n'

# To visualize the kernel we call its plot() function
fig, ax = plt.subplots(1, 1, figsize=fs)
k.plot(ax=ax)

# <markdowncell>

# Let's have a look at how the 'lengthscale' parameter changes our kernel. The parameters are accessed and changed using regex as follows:
# 
#     k['.*hp'] = 1.0
# 
# After running the cell below, change it to show how the variance affects the kernel.

# <codecell>

k = GPy.kern.RBF(1) 

fig, ax = plt.subplots(1, 1, figsize=fs)

theta = np.asarray([0.2,0.5,1.,2.,4.])
for t in theta:
    k['.*lengthscale']=t
    k.plot(ax=ax)
plt.legend(theta)

# <markdowncell>

# Any function that is positive semi-definite (PSD), i.e. the matrix evaluating the function at the input locations is PSD, is a valid kernel. The sum as well as the product of PSD functions is also PSD, so they are in turn also valid kernels.

# <codecell>

# We can add kernels
k1 = GPy.kern.RBF(1, lengthscale=2., variance=1.0)
k2 = GPy.kern.Cosine(1, lengthscale=0.3, variance=0.3)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=fs)
k1.plot(ax=ax1)
k2.plot(ax=ax2)

fig, (ax3, ax4) = plt.subplots(1, 2, figsize=fs)
k_add = k1 + k2
# print k_add, '\n\n'
k_add.plot(ax=ax3)

k_mult = k1 * k2
# print k_mult, '\n\n'
k_mult.plot(ax=ax4)


# <markdowncell>

# Many kernels are already implemented in GPy. The kernels that have been introduced so far are stationary (i.e. are dependent on the distance $|x_1 - x_2|$ rather than the actual values of $x_1$ or $x_2$). There are non-stationary kernels, e.g. the Brownian kernel.
# 
# If we want to access the covariance matrix of a kernel we can use the function
# 
#     k.K(x1, x2)
#     
# Let's have a look at the covariance matrices of some kernels

# <codecell>

X = np.linspace(0, 10).reshape((-1, 1))  # We need column vectors

cov_mat1 = k1.K(X, X)
cov_mat2 = k2.K(X, X)

# cov_mat_add = k_add.K(X, X)
cov_mat_prod = k_mult.K(X, X)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=fs)
ax1.imshow(cov_mat1, interpolation="nearest")
ax1.set_title('RBF kernel')
ax2.imshow(cov_mat2, interpolation="nearest")
ax2.set_title('Cosine kernel')

plt.figure(figsize=fs)
plt.imshow(cov_mat_prod, interpolation="nearest")
plt.title('Cosine * RBF kernel')

# <markdowncell>

# ##Gaussian Processes
# 
# We will start by drawing some sample functions from covariance matrices. Choosing a kernel or a combination of kernels is how we introduce our prior knowledge into the model, so understanding the properties of kernels qualitatively will be useful.
# 
# ###Draws from a GP kernel
# The prior probability distribution is given here again as a reminder:
# \begin{equation}
# \mathbf{f} \sim \mathcal{GP}(0, k(\mathbf{x},\mathbf{x})).
# \end{equation}
# We see that drawing a function from a GP prior is equivalent to sampling a vector from a multivariate Gaussian distribution defined by the mean function and the covariance matrix of our kernel.
# 
# We start with the RBF kernel, but feel free to replace this with any of the commented kernels or any other kernel you choose.

# <codecell>

num_draws = 5  # Number of functions to draw from the kernel
X = np.linspace(-5, 5, 100)[:, None]

k = GPy.kern.RBF(1, lengthscale=1.5, variance=1.)
# k = GPy.kern.Matern32(1, lengthscale=0.5, variance=1.)
# k = GPy.kern.PeriodicExponential(1, lengthscale=0.5, variance=1.)
# k = GPy.kern.Cosine(1, lengthscale=0.5, variance=1.)

# eps added along the diagonal for numerical stability. 
# If you get warnings about the covariance not being 
# positive definite, increase this value. 
eps = 5e-6

mean_vec = np.zeros((len(X),))
cov_mat = k.K(X, X) + eps * np.eye(len(X))

plt.figure(figsize=fs)
for ii in xrange(num_draws):
    f = np.random.multivariate_normal(mean_vec, cov_mat)
    plt.plot(X, f)
    
plt.xlim((min(X), max(X)))
plt.title('Sample functions from the kernel')

# <markdowncell>

# ###The GP Regression model
# Now we move to performing regression with a GP. The available models in GPy are accessible through the GPy.models module. 
# 
# The model we are interested in is the GPRegression model. This model is a GP prior with a Gaussian likelihood. We need to provide the training data (X, y) and the kernel to initialise the model.

# <codecell>

num_train = 30
num_test = 30

# <codecell>

# Create some toy data
eps = 5e-6
noise_var = 0.1  # Gaussian noise variance

X = np.linspace(-5, 5, num_train+num_test)[:, None]
k_data = GPy.kern.Matern32(1, lengthscale=1.5, variance=1.)
data_cov = k.K(X, X) + eps * np.eye(len(X))

# f = uncorrupted function; y = noisy observations
f = np.random.multivariate_normal(np.zeros(len(X)), data_cov)
f = (f - np.mean(f))/np.std(f)

y =  f + noise_var * np.random.randn(len(X))

# Split into train and test sets
idx = np.random.permutation(np.arange(num_train+num_test))
idx_train = np.sort(idx[:num_train])
idx_test = np.sort(idx[num_train:])

x_train = X[idx_train]
x_test = X[idx_test]
y_train = y[idx_train]
y_test = y[idx_test]

plt.figure(figsize=fs)
plt.title('Some toy data')
plt.plot(X, f, 'm')
plt.plot(x_train, y_train, 'c*', ms=ms)
plt.plot(x_test, y_test, 'y*', ms=ms)
plt.legend(['all data', 'training', 'testing'])

# <codecell>

# The printing and plotting interface is the same as we saw with kernels
k = GPy.kern.Matern32(1, lengthscale=1.0, variance=1.)
m = GPy.models.GPRegression(x_train, y_train[:, None], k)
print m,'\n\n'

# Plot the model prediction against the true curve
fig, ax = plt.subplots(1, 1, figsize=fs)
m.plot(ax=ax)
plt.plot(X, f, 'm')
plt.xlim((min(X), max(X)))
plt.title('GP regression model fit before optimization')

# <markdowncell>

# Optimizing a model is achieved by calling its optimize() function. 
# This performs a gradient search maximizing the log likelihood. 

# <codecell>

m.optimize()
print m,'\n\n'

# Plot the model prediction against the true curve
fig, ax = plt.subplots(1, 1, figsize=fs)
m.plot(ax=ax)
plt.plot(X, f, 'm')
plt.xlim((min(X), max(X)))
plt.title('Optimized GP regression model fit')

# <markdowncell>

# Try changing the true kernel (k_data), the model kernel (k) and its initial parameters. You will find that the choice of kernel type may influence the predictions significantly, so it is important to choose a suitable kernel.
# 
# ###Visualising the optimisation
# Let's plot the likelihood surface. This is a good way to track down irregularities between the optimiser and your own expectations, but only if the hyperparameter vector is small. We will look at the relationship between the log marginal likelihood (which is the optimiser's cost function) and the two kernel parameters. Note that such grid search quickly becomes infeasible even with medium-sized datasets and more than a couple of kernels.
# 
# In this example the likelihood surface exhibits a maximum (dark red area) close to the point(lengthscale, variance) = (3, 1), which confirms the optimiser's output.

# <codecell>

variances = np.linspace(0.5, 8., 20)
lengthscales = np.linspace(1., 9., 20)
log_likelihoods = np.zeros((len(variances), len(lengthscales)))

# m['Gaussian_noise.variance'] = 0.04
for ii in xrange(len(variances)):
    for jj in xrange(len(lengthscales)):
        m['Mat32.variance'] = variances[ii]
        m['Mat32.lengthscale'] = lengthscales[jj]
#         print m
        log_likelihoods[ii, jj] = m.log_likelihood()
        
plt.imshow(log_likelihoods, 
           extent=[min(lengthscales),max(lengthscales),min(variances),max(variances)], 
           aspect=1, 
           origin='lower')
plt.colorbar()

# <markdowncell>

# ### Higher-dimensional inputs
# 
# Gaussian Processes are not limited to single-dimensional inputs. Here is an example where our data is comprised of the sum of a sinusoid in x and a sinusoid in y.

# <codecell>

# Adapted from the GPy documentation

# sample inputs and outputs
X_2D = np.random.uniform(-3.,3.,(50,2))
y_2D = np.sin(X_2D[:,0:1])*np.sin(X_2D[:,1:2])+np.random.randn(50,1)*0.05

# define kernel
k_2D = GPy.kern.Matern52(2) + GPy.kern.White(2)

# create simple GP model
m_2D = GPy.models.GPRegression(X_2D, y_2D, k_2D)

# optimize and plot
m_2D.optimize(max_f_eval = 1000)

fig, ax = plt.subplots(1, 1, figsize=fs)
m_2D.plot(ax=ax)
print(m_2D)

# <markdowncell>

# ###Changing the model parameters manually
# There are a number of ways to change the model parameters. Accesssing them using regex is the preferred method.

# <codecell>

# Accessing the variable directly
# print m
m.kern.variance = 1.5
print m

# <codecell>

# Replacing the whole parameter vector
# NOTE: This will ignore any constraints that may have been set 
#       in the model. Use with care.
# print m
hps = m.param_array
hps *= 1.5
m[:] = hps
print m

# <codecell>

# Regex - preferred method
# print m
m['.*lengthscale'] = 2.
# There are two variances, so we need to be more specific
m['Gaussian_noise.variance'] = 1.5  
print m

# <markdowncell>

# ## Further reading
# 
# Here are some interesting papers and links on more specialised topics that may be of interest to some people. This is by no means an exhaustive list, but reflects some ideas I personally think may be interesting to the reader. 
# 
# ### Multi-output kernel regression
# GP regression is not limited to applications on single-output data sets, but can also be performed on multiple related sets of observations. Here is a great review paper on this type of model: http://arxiv.org/abs/1106.6251
# 
# The data set could take the form of prices of stocks in the same sector or measurements by co-located sensors. We expect the time series to behave similarly to some extent, so we use a kernel that not only describes the relationship between data points in a single time series, but also incorporates information the other time series bring to the table about the point(s) we are interested in predicting.
# 
# Multi-output GPs have been used in the field of geostatistics. Such models are referred to as coregionalization models. They refer to GP regression as (co-)kriging. An example application would be improving a predictive map of a difficult to measure pollutant. We measure its concentration on a relatively sparse grid and then perform inexpensive measurements of a different element (that can be used as a proxy for our pollutant) on a denser grid. We then fuse the two sets of data together to get a better prediction of the pollutant. 
# 
# The GPy documentation has a [tutorial on this topic](http://nbviewer.ipython.org/github/SheffieldML/notebook/blob/master/GPy/coregionalized_regression_tutorial.ipynb). 
# 
# ###Working with big data sets
# Gaussian Processes can be used very effectively for small to medium-sized data sets. Simple kernels and combinations of kernels allow us to encode prior information into our model very effectively. When we design the kernels to correspond to specific physical processes, then their optimal hyperparameter values may shed light onto the properties of the components, as well as giving us excellent predictive performance. See e.g. [this paper by Alvarez et. al.](http://jmlr.csail.mit.edu/proceedings/papers/v5/alvarez09a/alvarez09a.pdf) for a way to encode differential equations into a GP.
# 
# When the size of our data sets crosses roughly 10,000 data points, then the computational time of the matrix inversions in the log marginal likelihood (for training) and prediction equations become infeasible, due the inversion scaling as $O(N^3)$ in computation time and $O(N^2)$ in storage requirement. 
# 
# A number of different approaches are being explored to work around this limitation:
# * There are a number of sparse approximations to the full GP posterior that make certain assumptions about the nature of the data that give rise to computationally attractive properties. A nice summary to start at is [this review paper by Quinonero-Candela et. al.](http://jmlr.org/papers/volume6/quinonero-candela05a/quinonero-candela05a.pdf). Many sparse GP models are actually available in GPy out of the box. Here is a [tutorial on this topic by the authors of GPy](http://ml.dcs.shef.ac.uk/gpss/gpss13/labs/lab2.pdf).
# * Finding exactly equivalent models to the GP. For example the Matern kernels can be converted into Kalman Filters that return exactly the same results as the GP would but in a fraction of the time needed, as the matrices that inverted are much smaller. See [this paper by Hartikainen et. al.](http://www.lce.hut.fi/~ssarkka/pub/gp-ts-kfrts.pdf) for a discussion on this topic.
# * Choosing a subset of the data to train a standard GP. One of the directions that I am exploring is choosing a subset of data intelligently, e.g. by ranking data points according an [information theoretic criterion](http://arxiv.org/pdf/1112.5745.pdf) to ensure that the subset is maximally informative according to the criterion. One paper discussing such an application is [this one by Garnett et. al.](http://arxiv.org/abs/1310.6740)

# <codecell>


