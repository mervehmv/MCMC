# MCMC Experiments: Metropolis–Hastings & Langevin Monte Carlo

This notebook contains a set of small experiments exploring Markov Chain Monte Carlo (MCMC) methods, focusing on:

- Metropolis–Hastings (MH) sampling

- Unadjusted Langevin Algorithm (ULA) / Langevin MCMC in 1D

- A multivariate Langevin-style sampler for high-dimensional Gaussians

- Comparison of theoretical mixing time bounds (Theorems 3–5) as a function of dimension

The main goal is to illustrate how these algorithms sample from Gaussian target distributions, how step size and dimension affect convergence, and how empirical behavior relates to theoretical bounds.

## 1. Environment & Dependencies

Python version: 3.8+ (any recent version is fine)

Required packages:
```
numpy
scipy
matplotlib
seaborn
pandas
```

Install with:
```python
pip install numpy scipy matplotlib seaborn pandas
```

## 2. Notebook Structure

The notebook is organized into several parts:

### 2.1. Imports

The first cell imports the required libraries:
```python
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import wasserstein_distance
```

These provide random sampling, plotting, and probability density evaluations.

### 2.2. Metropolis–Hastings MCMC (Multivariate Gaussian)

This section implements Metropolis–Hastings to sample from a multivariate normal target distribution.

Key functions:

- target_distribution(x, mean, cov, dim, k=0):
  Evaluates the density of a multivariate normal distribution at x.
  For dim == 1, it uses the k-th coordinate of mean and cov.
  For higher dimensions, it uses mean and cov directly via scipy.stats.multivariate_normal.pdf.

- proposal_distribution(x, step_size, dim):
  Gaussian proposal distribution centered at the current state:
  np.random.multivariate_normal(x, np.diag([step_size] * dim))
  acceptance_probability(current_x, proposed_x, dim)

#### Computes the standard MH acceptance ratio:

$$
\alpha(x, x') = \min\left(1, \frac{\pi(x')}{\pi(x)}\right)
$$

where $\pi$ is the target density.

- metropolis_hastings(n_iterations, initial_x, step_size, dim)
  Runs the MH chain for n_iterations steps starting from initial_x.

Returns:

  samples – list of full dim-dimensional samples
  samples_x0 – first coordinate of each sample
  samples_x1 – second coordinate of each sample

Typical usage:
```python
n_iterations = 10000
dim = 2
initial_x = np.zeros(dim)
step_size = 0.5
mean = np.array([0, 2])
cov = np.array([[1, 0], [0, 5]])

samples, samples_x0, samples_x1 = metropolis_hastings(
    n_iterations, initial_x, step_size, dim
)
````

You can then visualize the marginal distributions of x[0] and x[1] using histograms and compare them to the true Gaussian density.

### 2.3. Langevin MCMC in 1D (Unadjusted Langevin Algorithm)

This part implements Langevin MCMC / ULA for a simple 1D target distribution.

- Target & gradient:
```python
# Target distribution (unnormalized)
def target_distribution(x):
    return np.exp(-x**2 / 2)  # (Gaussian; can be modified)

# Gradient of the log probability distribution
def log_prob_gradient(x):
    return x  # ∇ log π(x) for a standard normal
```
- Algorithm:
```python
def langevin_mcmc(num_samples, step_size):
    x_samples = np.zeros(num_samples)

    for i in range(1, num_samples):
        x_current = x_samples[i - 1]
        noise = np.random.normal(0, np.sqrt(2 * step_size))
        x_samples[i] = (
            x_current - step_size * log_prob_gradient(x_current) + noise
        )

    return x_samples
```
- Visualization:
```python
num_samples = 100000
step_size = 0.01
samples = langevin_mcmc(num_samples, step_size)
```

Plots the histogram of the second half of the chain vs. the true Gaussian density.
This illustrates how the ULA samples approximate the target.

### 2.4. High-Dimensional Langevin MCMC for Multivariate Gaussians

A more general Langevin-style sampler is implemented for a dim-dimensional Gaussian with mean mean and covariance cov.

- Log-gradient:
```python
def log_prob_gradient(x, mean, cov, dim):
    # For a multivariate normal N(mean, cov), ∇ log π(x) = Σ^{-1}(x - μ)
    gradient = np.linalg.solve(cov, (x - mean))
    return gradient
```
- Algorithm:
```python
def langevin_mcmc(n_iterations, step_size, mean, cov, dim, k=0):
    samples_x = []
    samples_x0 = []
    samples_x1 = []
    initial_x = np.zeros(dim)
    x_current = initial_x

    for i in range(1, n_iterations):
        noise = np.random.multivariate_normal(
            mean=np.zeros(dim), cov=np.identity(dim)
        )
        gradient = log_prob_gradient(x_current, mean, cov, dim)
        initial_x = (
            x_current - step_size * 0.5 * gradient + np.sqrt(step_size) * noise
        )

        samples_x.append(initial_x)
        samples_x0.append(initial_x[0])
        samples_x1.append(initial_x[1])
        x_current = initial_x

    return samples_x, samples_x0, samples_x1
```
- Typical usage:
```python
n_iterations = 10000
dim = 100
step_size = 0.05
mean = np.zeros(dim)
cov = np.identity(dim)

samples_x, samples_x0, samples_x1 = langevin_mcmc(
    n_iterations, step_size, mean, cov, dim
)
```

Histograms of samples_x0 and samples_x1 are then compared with the corresponding 1D Gaussian marginals to evaluate how well the sampler mixes in high dimensions. Plots for different numbers of iterations are created to show convergence behavior.

The notebook saves some of these plots (e.g., ula.png).

### 2.5. Theoretical Bounds: Theorems 3–5

The final section defines three helper functions, theorem3, theorem4, and theorem5, which compute logarithms of iteration bounds as functions of the dimension p and accuracy parameter ε.

These represent different theoretical upper bounds on the number of iterations needed for convergence in high-dimensional settings.

```python
def theorem3(p, epsilon):
    M = 20
    m = 10
    return np.log((1.65**2) * (M**2) * p / ((m**2) * (epsilon**2)))

def theorem4(p, epsilon):
    M = 20
    m = 10
    K1 = np.ceil(
        np.log((p + p/m) / np.sqrt(p))
        + np.log(m / M)
        + 0.5 * np.log(M + m)
    ) / np.log(1 + 2 * m / (M - m))
    return np.log(
        ((3.5**2) * (M**2) * p / ((epsilon**2) * (m**2)) - M - m) * 3 / (2 * m)
        + K1
    )

def theorem5(p, epsilon):
    M = 20
    m = 10
    return np.log(
        (4 * M * p * ((m + M)**2) / m)
        * (1 + 1 / ((p + p/m)**2))
        / (m * M * (epsilon**2))
    )

```
The notebook plots these bounds against dimension:

```python
p = np.linspace(16, 1000, 100)
plt.plot(p, theorem5(p, 0.001), ..., label='ε = 0.001 DM bound')
plt.plot(p, theorem3(p, 0.001), ..., label='ε = 0.001 Theorem 1')
plt.plot(p, theorem4(p, 0.001), ..., label='ε = 0.001 Theorem 2')
# (and similarly for ε = 0.005 and ε = 0.02)
```

The resulting figure (graph_paper.png) visualizes how different theoretical guarantees scale with dimension and target accuracy.

## 3. How to Use / Modify the Notebook

- Change the target distribution:
    Adjust mean, cov, or the 1D target_distribution to experiment with other distributions (e.g., mixtures, anisotropic Gaussians).

- Adjust algorithm parameters:
    n_iterations / num_samples – affects how long the chain runs.
    step_size - crucial for stability vs. mixing; too small mixes slowly, too large can lead to poor approximation.
    dim - controls the dimensionality of the multivariate experiments.

- Compare empirical vs. theoretical behavior:
    Use the high-dimensional Langevin sampler along with the Theorem 3–5 plots to build intuition about how dimension and ε affect the number of required iterations.
