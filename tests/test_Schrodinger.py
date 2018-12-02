#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import tensorflow as tf
import math
import numpy as np
import argparse

tf.enable_eager_execution()

class basis_unit_tests(unittest.TestCase):
    # Tests if first basis vector (1) is returned
    def test_first(self):
        self.assertEqual(basis(0, 0), 1)
        self.assertEqual(basis(0, 1), 1)
        
    # Tests if sin basis vectors are returned
    def test_sin(self):
        self.assertEqual(0, basis(1, 0))
        self.assertEqual(basis(1, 1), math.sin(1))
        self.assertEqual(basis(3, 1), math.sin(2))
        
    # Tests if cos basis vectors are returned
    def test_cos(self):
        self.assertEqual(1, basis(2, 0))
        self.assertEqual(math.cos(1), basis(2, 1))
        self.assertEqual(math.cos(2), basis(4, 1))

class trapz_unit_tests(unittest.TestCase):
    # Tests if trapz integrates simple function properly
    def test_integrate(self):
        def fun(x):
            return x

        xdata = [0, 1]
        self.assertEqual(1/2, trapz(fun, xdata))
        
        xdata2 = [1, 2]
        self.assertEqual(3/2, trapz(fun, xdata2))

class data_unit_tests(unittest.TestCase):
    # Tests if x data is opened properly
    def test_x(self):
        d = data('potential_energy.dat', 5)
        self.assertEqual(d.x, [0, 1.57079, 3.14159, 4.71238, 6.28318, 7.85398, 9.42477])
    
    # Tests if potential energy data is opened properly
    def test_V(self):
        d = data('potential_energy.dat', 5)
        self.assertEqual(d.V, [0, 6, 0, -6, 0, 6, 0])
    
    # Tests if basis_size is stored properly
    def test_basis_size(self):
        d = data('potential_energy.dat', 5)
        self.assertEqual(d.basis_size, 5)
    
    # Tests if H matrix is initialized properly
    def test_H(self):
        d = data('potential_energy.dat', 5)
        tf.assert_equal(d.H, tf.zeros([5, 5], dtype = tf.float64))