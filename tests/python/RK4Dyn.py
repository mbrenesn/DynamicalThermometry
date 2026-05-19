import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

import os 
import sys

sys.path.append(os.path.abspath("../"))

#=========================================================================================
# ODE system to solved at different times
# dv_dt = Qv with v(t=0) = v_0
# dw_dt = Qw + Rv with w(t=0) = w_0
#=========================================================================================
def gen_ode(Q, R, dim):
    def ode_sys(t, y): 
        v = y[0:dim]
        w = y[dim:]
        dv_t = np.dot(Q,v)
        dw_t = np.dot(Q, w) + np.dot(R, v)
        return np.concatenate([dv_t, dw_t])

    return ode_sys

#=========================================================================================
# Relative error between a state at the time t solved with RK4 (C++) vs solve_ivp (python)
#=========================================================================================
def metric(w_ivp, w_rk4):
    return np.linalg.norm(w_ivp - w_rk4) / np.linalg.norm(w_ivp)

#=========================================================================================
# Reading the data for the operators and transforming to numpy
#=========================================================================================
dfQ = pd.read_csv('../data/Q.csv', sep=' ', header=None)
dfR = pd.read_csv('../data/R.csv', sep=' ',header=None)
Q = dfQ.to_numpy(dtype=np.complex128)
R = dfR.to_numpy(dtype=np.complex128)

#=========================================================================================
# Reading the initial states for w(t), v(t) and the time evaluations. 
# Then they are transform to numpy arrays of the require shape (row vectors)
#=========================================================================================
dv0 = pd.read_csv('../data/v0.csv', sep=' ', header=None)
dw0 = pd.read_csv('../data/w0.csv', sep=' ', header=None)
w0 = dw0.to_numpy(dtype=np.complex128)
v0 = dv0.to_numpy(dtype=np.complex128)
w0 = w0.reshape(w0.size) #pandas reads column vectors, the must be row vectors.
v0 = v0.reshape(v0.size)
dt_eval = pd.read_csv('../data/time_eval.csv', sep=' ', header=None)
t_eval = dt_eval.to_numpy(dtype=np.float64)
t_eval = t_eval.reshape(t_eval.size)


#=========================================================================================
# Reading the file with all the solutions stacked in rows as a matrix
#=========================================================================================
dw_cpp = pd.read_csv("../data/sol.csv", sep=' ', header=None)
w_cpp = dw_cpp.to_numpy(dtype=np.complex128) 

#=========================================================================================
# Dynamics solution using solve_ivp
#=========================================================================================
dim = w0.size
y0 = np.concatenate([v0, w0]) #initial conditions
t_span_sys = (0, t_eval.max())

ode_func = gen_ode(Q, R, dim)

sol = solve_ivp(ode_func, t_span_sys, y0, t_eval=t_eval, rtol=1e-13, atol=1e-16) #rtol and atol help with the step size

w_ivp = sol.y.T[:,dim:] #Evolved states in the same layout as w_cpp
error = np.array([metric(w_ivp[i, :], w_cpp[i, :]) for i in range(w_cpp.shape[0])]) #array that contains relative error by time.

print("="*78)
print("ODE DYNAMICS SOLVER COMPARISON RK4(C++) AND SOLVE_IVP (Python) for the system: ")
print("                   dv_dt = Qv with v(t=0) = v_0                                ")
print("                   dw_dt = Qw + Rv with w(t = 0) = w_0")
print("="*78)
print("ANALYSIS")
print("="*78)
print(f"Average error between methods: {error.mean():.3e}")
print(f"Standart deviation of error between methods: {error.std():.3e}")
print(f"Maximum error between methods: {error.max():.3e}")
