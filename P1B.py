import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, poisson, binom, expon

mean = 50
std_dev = 10


NORMAL_DISTRO = lambda x, mean, std_dev: norm.pdf(x, mean, std_dev)
EXPONENTIAL_DISTRO = lambda x, scale: expon.pdf(x, scale=scale)


def draw_graph(sample, color, labels, linespace_data, distro_graph=1, figsize=(8, 6)):
    plt.figure(figsize=figsize)
    plt.hist(sample, bins=30, density=True, alpha=0.6, color=color)
    x = np.linspace(*linespace_data)
    if distro_graph:
        plt.plot(
            x, NORMAL_DISTRO(x, mean, std_dev),'r-',lw=2, label="Normal Distribution"
        )
    else:
        plt.plot(
            x, EXPONENTIAL_DISTRO(x, 2), "r-", lw=2, label="Exponential Distribution"
        )
    plt.title(labels[0])
    plt.xlabel(labels[1])
    plt.ylabel(labels[2])
    plt.legend()
    plt.grid(True)
    plt.show()


# Normal Distro
sample = np.random.normal(mean, std_dev, 1000)
draw_graph(
    sample,
    linespace_data=(mean - 4 * std_dev, mean + 4 * std_dev, 100),
    color="blue",
    labels=[
        "Normal Distribution Example (Quality Control)",
        "Values",
        "Probability Density",
    ],
)


# Poisson Distribution
lambda_param = 5
k = 3
prob_3_events = poisson.pmf(k, lambda_param)
print(f"\n\nThe Probability of 3 events occuring in an hours: {prob_3_events}")

# Binomial Distribution
n = 10
p = 0.6
k_success = 7
prob_7_success = binom.pmf(k_success, n, p)
print(f"\n\nProbability of 7 success out of 10 trials: {prob_7_success}")

# Exponential Distribution

exp_sample = np.random.exponential(scale=2, size=1000)
draw_graph(
    exp_sample,
    color="green",
    labels=["Exponential Distribution Example", "Values", "Probability Density"],
    linespace_data=(0, 10, 100),
    distro_graph=0,
)
