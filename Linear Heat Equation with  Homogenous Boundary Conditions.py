import numpy as np
from numba import jit, int32, float32, prange
from scipy.sparse import diags
from scipy.integrate import RK45, solve_ivp
from scipy.sparse.linalg import inv
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib
from time import time
import pandas as pd
import plotly.express as px
import time
import os
import math


class Heat:

    #A *alpha - beta(D *) +beta(E(1 - 0)alpha) = 0

    def __init__(self, N=10):

        self.N = N
        self.H = 1/N
        self.domain = np.linspace(0, 1, N+1)  # 0 - 1
        self.beta = -1
        self.t = 1
        self.rate = self.H
        #Set initial conditions here
        self.alpha = np.sin(np.pi * self.domain)[1:-1]
        #Set solution here
        self.exact = lambda t: math.exp(t) * self.alpha
        #Define boundary conditions
        self.b_c_start = lambda t: 0
        self.b_c_end = lambda t: 0
        #Define f

        def f(k):
            def neg(k): return (-(math.pi**2*self.beta - 1)*(math.pi*self.H*math.cos(math.pi*self.H*k) +
                                                             math.sin(math.pi*self.H*k) - math.sin(math.pi*self.H*k + math.pi*self.H))/(math.pi**2*self.H))
            def pos(k): return ((math.pi**2*self.beta - 1)*(math.pi*self.H*math.cos(math.pi*self.H*k) -
                                                            math.sin(math.pi*self.H*k) + math.sin(math.pi*self.H*k - math.pi*self.H))/(math.pi**2*self.H))
            if k == 0:
                return neg(k)
            elif k == N:
                return pos(k)
            else:
                return neg(k) + pos(k)



        self.A = (self.H) * sp.csc_matrix(diags([[1/6 for i in range(N)],
                                                 [1/3] + [2/3]*(N-1) + [1/3],
                                                 [1/6 for i in range(N)]],
                                                [1, 0, -1]), dtype=np.float64)[1:-1, 1:-1]
        self.inv_A = inv(self.A)
        print("Inverse A   ", self.inv_A)
        self.D = N * sp.csr_matrix(diags([[-1 for i in range(N)],
                                          [1] + [2]*(N-1) + [1],
                                          [-1 for i in range(N)]],
                                         [1, 0, -1]), dtype=np.float64)
        self.D[0, 0] -= N
        self.D[0, 1] -= -N
        self.D[N, N-1] -= -N
        self.D[N, N] -= N
        self.D = self.D[1:-1, 1:-1]
        print("D   ", self.D)
        self.F = np.array([f(k) for k in prange(N+1)])[1: -
                                                       1].reshape(-1, 1)  # incomplete without exp(t)
        print("F   ", np.exp(1)*self.F)

        self.hj = inv(self.A - self.rate*self.beta*self.D)

    def error(self):
        return np.linalg.norm(self.exact(self.t) - self.approx, 2)

    def func(self, t,  y):
        d = self.inv_A.dot(self.beta*self.D.dot(y) + math.exp(t)*self.F)
        return d

#     def fun(self, t, y):
#         return (self.A).dot(self.beta*self.D.dot(y) + math.exp(t)*self.F  )

    def algo_1(self):

        self.algo = solve_ivp(self.fun, t_span=(0, self.t),  y0=self.alpha,
                              t_eval=np.array([self.t]), vectorized=True, method="LSODA")
        self.approx = self.algo.y.reshape(-1, )

    def algo_2(self):

        # Finds value of y for a given x using step size h
        # and initial value y0 at x0.
        def backward_euler(y):
            t = 0
            while t < self.t:
                "Apply Runge Kutta Formulas to find next value of y"

                # Update next value of y
                y = self.hj.dot(self.A.dot(y) + np.exp(t) * self.F * self.rate)

                # Update next value of x
                t += self.rate

            return y

        self.algo = backward_euler(self.alpha.reshape(-1, 1))
        self.approx = self.algo.reshape(-1, )

    def algo_3(self):

        def rungeKutta(y):
            t = 0
            while t != self.t:
                k1 = self.H * self.func(t, y)
                k2 = self.H * self.func(t + 0.5 * self.H, y + 0.5 * k1)
                k3 = self.H * self.func(t + 0.5 * self.H, y + 0.5 * k2)
                k4 = self.H * self.func(t + self.H, y + k3)

                # Update next value of y
                y += (1.0 / 6.0)*(k1 + 2 * k2 + 2 * k3 + k4)

                # Update next value of x
                t += self.H
            return y

        self.algo = rungeKutta(self.alpha.reshape(-1, 1))
        self.approx = self.algo.reshape(-1, )


obj = Heat(8)
