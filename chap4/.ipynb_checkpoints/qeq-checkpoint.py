import numpy as np

def qeq(a, b, c):
    d = np.sqrt(b**2 - 4 * a * c)
    return ((-b + d) / (2 * a), (-b - d) / (2 * a))


def qeq2(a, b, c):
    alpha = (-b - np.sign(b) * np.sqrt(b**2 - 4 * a * c)) / (2 * a)
    beta = c / (a * alpha)
    return (alpha, beta)