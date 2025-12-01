import numpy as np
import pandas as pd
import pycirce as pyc
import CoolProp as cp
import random
import copy
import jax.numpy as jnp
from kernax.kernels import Energy
from kernax import KernelHerding
import matplotlib.pyplot as plt 
from matplotlib import rc
import seaborn as sns

rc('text', usetex=True)
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 22})
rc('lines', linewidth=3)
rougeCEA = "#b81420"
orangeEDF = "#fe5716"
bleuEDF = "#10367a"

def blasius(x):
    """
    Helper function that computes the shear coefficient using Blasius correlation
    """

    P = x[:, 0]
    G = x[:, 1]
    T = x[:, 3]
    D = x[:, 4]

    lambda_f = cp.CoolProp.PropsSI('L', 'T', T, 'P', P, 'HEOS::Water') * 1e-3
    
    return np.log10(0.316) - 0.25 * np.log10((G * D) / lambda_f)

