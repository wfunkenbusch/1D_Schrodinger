# -*- coding: utf-8 -*-

import tensorflow as tf
import math
import numpy as np
import argparse

tf.enable_eager_execution()

def basis(i, x):
    '''
    Returns basis vector i evaluated at x. Basis vectors are (starting at i = 0):
    1, sin(x), cos(x), sin(2x), cos(2x), ....
        
    Arguments:
        i (integer) - 
            The index of the desired basis vector.
        x (float) - 
            The value at which the basis vector is to be evaluated.
    
    Returns:
        val (float) - 
            The value of the basis vector evaluated at x.    
    '''
    if i % 2 == 0:
        return math.cos(i // 2 * x)
    else:
        return math.sin((i + 1) // 2 * x)

def trapz(fun, xdata):
    '''
    Integrates a function with respect to xdata using trapezoidal integration.
    
    Arguments:
        fun (function) - 
            The function to be integrated, with x as its first argument.
        
        xdata (list) - 
            The x data to be integrated over.
            
    Returns:
        integral (float) - 
            The value of the trapezoidal integration.
    '''
    integral = 0
    
    for i in range(len(xdata) - 1):
        # Area at each iteration is 1/2 (x[i + 1] - x[i]) * (y(x[i + 1]) + y(x[i]))
        dx = xdata[i + 1] - xdata[i]
        ysum = fun(xdata[i + 1]) + fun(xdata[i])
        integral += 1/2*ysum*dx
        
    return integral