# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 11:13:10 2020

@author: fanyak
"""
import numpy as np;
from sympy import Symbol

# =============================================================================
# Y = np.array([
#         [5, 0, 7],
#         [0,2,0],
#         [4,0,0],
#         [0,3,6],
#     ]);
# 
# rows, cols = Y.shape;
# X = np.zeros([rows, cols]);
# a = Symbol('a')
# u = np.array([[6,0,3,6]]); 
# v = np.array([[4,2,1]]);
# l = 1;
# 
# # compute matrix X
# X = np.matmul(u.T, v); 
# 
# def error(Y, X):
#     squared_error = 0;
#     for a in range(rows):
#         for i in range(cols):
#             if(Y[a][i] != 0):
#              squared_error += (Y[a][i] - X[a][i])**2;
#     return squared_error / 2
# 
# 
# def regularization(l, u, v):
#     lu = (l/2)*(u * u).sum()
#     lv =  (l/2)*(v * v).sum();
#     return  lu + lv;
# =============================================================================


x = np.arange(1,7);
A = np.array([
        [1/6, 1/6, 1/6, 1/6 , 1/6, 1/6],
        [1/3, 1/3, 1/3, -1/3 , -1/3, -1/3]
        ])

