import numpy as np

def increasing_exp(x, a, b, c):  # "warm"
    return a*(1-np.exp(b*(x-c))) #(a+b*np.exp(c*(x-d)))

def increasing_double_exp(x, a, b, c, d, f, g):  
    return a*(1-np.exp(b*(x-c))) + d*np.exp(f*(x-g))

def decreasing_exp(x, a, b, c):  # "cold"
    return a*np.exp(b*(x-c))