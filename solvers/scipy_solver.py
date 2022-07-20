
import abc
import math

import numpy as np
from scipy.optimize import minimize, NonlinearConstraint


class SolverBase(metaclass=abc.ABCMeta):
    def __init__(self, n_vars, a=None, eps=None, grad=None):
        self.n_vars = n_vars
        self.a = a
        self.eps = eps
        self.grad = grad
        self.sol = None
        self.r = [1.0,] * len(self.a)

    
    def set_range(self, _range):
        self.r[:] = _range[:]

    @abc.abstractmethod
    def run(self):
        pass

class ScipySolver(SolverBase):
    def __init__(self, n_vars, a=None, eps=None, grad=None):
        super(ScipySolver, self).__init__(n_vars, a, eps, grad)
        self.lb = 1e-32
        self.bounds = [(1e-23, 1) for _ in range(self.n_vars)]
        # self.grad = self.grad.sum(axis=1)

    def run(self):
        # lc = LinearConstraint(self.grad.reshape(1, self.n_vars), np.zeros(1),
                                #  np.array([self.eps**2]))
        nlc = NonlinearConstraint(fun=self.con, lb=0.0, ub=self.eps**2, jac=self.con_jac)
        x0 = np.ones(self.n_vars) * (self.eps**2/self.n_vars) / (self.grad)
        # x0 = np.ones(self.n_vars) * 0.00001
        x0 = np.sqrt(x0)
        # print(f'x0: {x0}')
        # exit(-1)
        res = minimize(self.obj, x0, method='SLSQP',  constraints=nlc, bounds=self.bounds, options={'maxiter': 10000})
        # res = minimize(self.obj, x0, method='trust-constr', jac=self.jac, constraints=nlc, bounds=self.bounds, options={'maxiter': 100000})
        self.sol = np.array(res.x)
        # print(f'sol: {self.sol}')
        print(res)
        return res

    def con(self, x):
        # print(f'x: {x}')
        g = self.grad 
        g = g.reshape(1, self.n_vars)
        x_ = x * x
        c = np.dot(g, x_)
        print(f'con: {c}')
        return c
    
    def con_jac(self, x):
        # g = self.grad.reshape(1, self.n_vars)
        j = 2 * self.grad * x
        return j

    def obj(self, x):
        # return np.dot(-np.log2(x), self.a)
        f = np.dot(-np.log2(np.abs(x)), self.a)
        # f = x.dot(x)
        return f

    def jac(self, x):
        return ((x + self.lb) ** -1) * (-1.0/math.log(2)) * self.a

    def get_bits(self):
        bits = []
        for i in range(self.n_vars):
            bits.append(np.ceil((-math.log2(np.abs(math.sqrt(3)*self.sol[i]/self.r[i])))))
            # bits.append(np.ceil(-math.log2(self.sol[i])/2) - 1)
        return bits



class ScipySolverMemMode(SolverBase):
    def __init__(self, n_vars, a=None, eps=None, grad=None):
        super(ScipySolverMemMode, self).__init__(n_vars, a, eps, grad)
        self.lb = 1e-32
        self.bounds = [(self.lb, 1) for _ in range(self.n_vars)]
        # self.grad = self.grad.sum(axis=1)

    def run(self):
        # self.grad /= self.grad.min()
        nlc = NonlinearConstraint(fun=self.con, lb=(0,), ub=(self.eps*self.n_vars*32, ), jac=self.con_jac)
        x0 = np.ones(self.n_vars) * (self.eps**2/self.n_vars) / (self.grad)
        # x0 = np.ones(self.n_vars) * (1/self.n_vars) / (self.grad)
        # x0 = np.ones(self.n_vars) * (1.0 / math.sqrt(3) / 2**(self.eps*32))
        # x0 = x0 * self.r
        # print(x0)
        # x0 = np.ones(self.n_vars) * 1e-10
        # res = minimize(self.obj, x0, jac=self.jac, method='COBYLA', constraints=nlc, bounds=self.bounds, options={'maxiter': 10000})
        res = minimize(self.obj, x0, jac=self.jac, method='SLSQP', constraints=nlc, bounds=self.bounds, options={'maxiter': 10000})
        # res = minimize(self.obj, x0, jac=self.jac, method='trust-constr', constraints=nlc, hess=self.hess, bounds=self.bounds, options={'maxiter': 10000})
        print(res)
        self.sol = np.array(res.x)
        print(f'sol: {self.sol}')
        return res

    def con(self, x):
        # print('x: ', x)
        c = -np.log2(math.sqrt(3) * np.abs(x+1e-23) / self.r)
        # print('eval con: ', sum(c))
        return np.sum(c)
    
    def con_jac(self, x):
        j = ((x+self.lb)**-1) * (-1.0/math.log(2)) 
        # print('eval con_jac: ', j)
        return j

    def obj(self, x):
        return 0.5 * np.dot(self.grad, x**2)

    def jac(self, x):
        # return (x ** -1) * (-1.0/math.log(2)) * self.a
        j = self.grad * x# - self.lb)
        # print('x:', x)
        # print('j:', j)
        return j

    def hess(self, x):
        h = np.diag(self.grad)
        return h

    def get_bits(self):
        bits = []
        for i in range(self.n_vars):
            bits.append(np.ceil((-math.log2(self.lb + np.abs(math.sqrt(3)*(self.sol[i]))/self.r[i]))))
            # bits.append(np.ceil(-math.log2(self.sol[i])/2) - 1)
        return bits