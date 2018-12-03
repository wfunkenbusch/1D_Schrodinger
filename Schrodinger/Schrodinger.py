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
            Scaling constant for the kinetic energy term. Must be positive. Default 1.
            
        self.domain (list) - 
            Contains the minimum and maximum value of the domain. Note that the potential energy term will only
            be calculated for the range given in the imported file (it will be treated as 0 elsewhere).
            Default [0, 9.42477]

        self.H (tensor) - 
            Hamiltonian matrix whose eigenvalues and eigenvectors solve the Schrodinger equation.
            
        self.min_e (tensor) - 
            The minimum energy value. Must be positive
            
        self.min_v (tensor) - 
            The eigenvector corresponding to the minimum energy value (self.min_e). The elements correspond to the
            coefficients of a basis function.
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
        -c hbar^2 d^2f/dx^2. For the Fourier basis, the second derivative of sin(ax) is -a^2 sin(ax), so the term is 
        a^2 c hbar^2 sin(ax) (or cos(ax)). In reduced units, hbar = 1. In the Hamiltonian matrix, the kinetic energy part
        of H[i][j] is the inner product of the Hamiltonian applied to the ith basis element with the jth basis element.
        '''
        # Range of x across the input domain
        xrange = self.domain[0] + range(1000) * (self.domain[1] - self.domain[0]) / 1000
        for i in range(self.basis_size):
            for j in range(self.basis_size):
                self.H = tf.scatter_nd_add(self.H,
                                           tf.constant([[i, j]]),
                                           tf.constant([self.c * trapz(lambda x: ((i + 1) // 2)**2 * basis(i, x) * basis(j, x),
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

    def compute_coefficients(self):
        '''
        Computes minimum energy level and the corresponding coefficients for the basis set. Minimum energy level must
        be positive.
        '''
        # Computes eigenvalues and eigenvectors
        e, v = tf.linalg.eigh(self.H)
        
        # Stores minimum energy level
        self.min_e = e[0]
        self.min_index = 0
        
        for i in range(1, e.shape[0]):
            # Energy level must be positive
            if tf.math.greater(0, self.min_e):
                self.min_e = e[i]
                self.min_index = i
            # Replace old minimum energy level only if the new one is positive and less than the old
            if tf.math.greater(e[i], 0) and tf.math.greater(self.min_e, e[i]):
                self.min_e = e[i]
                self.min_index = i
                
        self.min_v = v[self.min_index]
        
        print('Minimum energy level: {}' .format(self.min_e))
        print('Coefficients for Fourier basis: {}' .format(self.min_v))

def get_parser():
    '''
    Allows for user input from the command line.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--FileName', type = str, default = 'd', help = 'String: Base file name')
    parser.add_argument('--basis_size', type = int, default = 5, help = 'Integer: Number of basis functions')
    parser.add_argument('--c', type = float, default = 1, help = 'Float: Scaling constant for the kinetic energy term')
    parser.add_argument('--domain', type = list, default = [0, 9.42477], help = 'Bounds for kinetic energy domain')
    
    args, unknown = parser.parse_known_args()

    return args
        
def main():
    '''
    Main function. Computes and reports minimum energy value and the corresponding basis function coefficients.
    '''
    args = get_parser()
    d = data(args.FileName, args.basis_size, args.c, args.domain)
    d.kinetic_energy()
    d.potential_energy()
    d.compute_coefficients()    
        
if __name__ == '__main__':
    main()
