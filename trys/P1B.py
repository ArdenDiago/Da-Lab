from matplotlib.pylab import sca
import numpy as np
from scipy.stats import norm, binom, poisson, expon
import matplotlib.pyplot as plt

mean = 50
std = 10

NORMAL_dISTOR = lambda x, mean, std: norm.pdf(x, mean, std)
EXPO_DISTRO = lambda x, scale: expon.pdf(x, scale=scale)


def draw_graph(sample, color, lables, data, graph=1):
    plt.figure(figsize=(9, 8))
    plt.hist(sample, bins=30, density=True, alpha=0.6, color=color)
    x = np.linspace(*data)
    if graph:
        plt.plot(
            x,
            NORMAL_dISTOR(x, mean, std),
            "r-",
            lw=2,
            color="red",
            label="NormalDistor",
        )
    else:
        plt.plot(x, EXPO_DISTRO(x, 2), "r-", lw=2, label="Expo")
    plt.title(lables[0])
    plt.xlabel(lables[1])
    plt.ylabel(lables[2])
    plt.legend()
    plt.grid(True)
    plt.show()


# Normal
sample = np.random.normal(mean, std, 1000)
draw_graph(
    sample,
    "blue",
    ["Normal", "Value", "Prob Density"],
    data=(mean - 4 * std, mean + 4 * std, 100),
)

sample = np.random.exponential(scale=2, size=1000)
draw_graph(
    sample, "green", ["Exp Gaph", "value", "prob Density"], data=(0, 10, 100), graph=0
)

# piosn
sp = 10
e = 3
prob_3 = poisson.pmf(e, sp)

# Binom
sp = 10
prob = 0.6
success = 7
prob_7 = binom.pmf(success, sp, prob)
