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

class data:
    '''
    Stores potential energy data and performs operations.
    
    Keys: 
        self.x (list) - 
            x data.
            
        self.V (list) -
            Potential energy data.
            
        self.basis_size (integer) - 
            Desired size of basis set. Basis functions come from the Fourier series.
            
        self.c (float) - 
            Scaling constant for the kinetic energy term. Default 1.
            
        self.domain (list) - 
            Contains the minimum and maximum value of the domain. Note that the potential energy term will only
            be calculated for the range given in the imported file. Default [0, 9.42477]
    '''
    def __init__(self, FileName, basis_size, c = 1, domain = [0.0, 9.42477]):
        '''
        Stores x and potential energy data.
        
        Arguments:
            FileName (string) - 
                Path of the file.
                
            basis_size (integer) - 
                Desired size of basis set. Basis functions come from the Fourier series.
                
            c (float) - 
                Scaling constant for the kinetic energy term.
                
            domain (list) - 
                Contains the minimum and maximum value of the domain. Note that the potential energy term will only
                be calculated for the range given in the imported file.
        '''
        # Opens file and removes first line (header)
        file_data = np.loadtxt(FileName)
        self.x = list(file_data[:, 0])
        self.V = list(file_data[:, 1])
        
        # Stores basis_size, scaling constant, and domain
        self.basis_size = basis_size
        self.c = c
        self.domain = list(np.array(domain))
        
        # Creates tensor which will store Hamiltonian matrix.
        self.H = tf.Variable(tf.zeros([self.basis_size, self.basis_size], dtype = tf.float64))

    def kinetic_energy(self):
        '''
        Adds kinetic energy term to Hamiltonian matrix. The kinetic energy term in the Hamiltonian is
        -hbar^2 d^2f/dx^2. For the Fourier basis, the second derivative of sin(ax) is -a^2 sin(ax), so the term is 
        a^2 hbar^2 sin(ax) (or cos(ax)). In reduced units, hbar = 1. In the Hamiltonian matrix, the kinetic energy part
        of H[i][j] is the inner product of the Hamiltonian applied to the ith basis element with the jth basis element.
        '''
        # Range of x across the input domain
        xrange = self.domain[0] + range(1000) * (self.domain[1] - self.domain[0]) / 1000
        for i in range(self.basis_size):
            for j in range(self.basis_size):
                self.H = tf.scatter_nd_add(self.H,
                                           tf.constant([[i, j]]),
                                           tf.constant([self.c*trapz(lambda x: ((i + 1) // 2)**2 * basis(i, x) * basis(j, x),
                                                                     xrange)], dtype = tf.float64))
                
    def potential_energy(self):
        '''
        Adds potential energy term to Hamiltonian matrix. The potential energy term in the Hamiltonian is the input
        data (self.V). In the Hamiltonian matrix, the potential energy part of H[i][j] is the inner product of the
        potential energy times the ith basis element with the jth basis element.
        '''
        for i in range(self.basis_size):
            for j in range(self.basis_size):
                self.H = tf.scatter_nd_add(self.H,
                                           tf.constant([[i, j]]),
                                           tf.constant([trapz(lambda x: self.V[self.x.index(x)] * basis(i, x) * basis(j, x),
                                                              self.x)], dtype = tf.float64))
