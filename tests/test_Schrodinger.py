#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import tensorflow as tf
import math
import numpy as np
import argparse

from Schrodinger.Schrodinger import *

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

    # Tests if KE values are calculated and stored properly
    def test_KE(self):
        d = data('potential_energy.dat', 5)
        d.kinetic_energy()
        tf.assert_equal(d.H[0, 0], tf.constant(0, dtype = tf.float64))
        # Integral of sin(x) from 0 to 3pi is 2
        tf.assert_equal(d.H[1, 0], tf.constant(1.9999407080507803, dtype = tf.float64))
        # Integral of cos(x) from 0 to 3pi is 0
        tf.assert_equal(d.H[2, 0], tf.constant(0.00943252106630788, dtype = tf.float64))
        # Integral of sin^2(x) from 0 to 3pi is 3pi/2 = 4.7
        tf.assert_equal(d.H[1, 1], tf.constant(4.712388560987876, dtype = tf.float64))
        # Integral of sin(x)cos(x) from 0 to 3pi is 0
        tf.assert_equal(d.H[1, 2], tf.constant(4.448556822280619e-05, dtype = tf.float64))
        # Integral of 4sin(2x)cos(x) from 0 to 3pi is 16/3
        tf.assert_equal(d.H[3, 2], tf.constant(5.3328590283266255, dtype = tf.float64))
    
    # Tests if scaling coefficient works properly
    def test_c(self):
        d = data('potential_energy.dat', 5, 2)
        d.kinetic_energy()
        # Twice the value of the previous
        tf.assert_equal(d.H[1, 0], tf.constant(2*1.9999407080507803, dtype = tf.float64))
    
    # Tests if domain input works properly
    def test_domain(self):
        d = data('potential_energy.dat', 5, 1, [0, 2*3.1415])
        d.kinetic_energy()
        # Integral of sin(x) from 0 to 2pi is 0
        tf.assert_equal(d.H[1, 0], tf.constant(2.0919357132000873e-05, dtype = tf.float64))
    
    # Tests if PE values are calculated and stored properly
    def test_PE(self):
        d = data('potential_energy.dat', 3)
        d.potential_energy()
        # Average value is 3pi
        tf.assert_equal(d.H[0, 0], tf.constant(9.424770000000002, dtype = tf.float64))
        # Integral of cos(x) from 0 to 3pi is 0
        tf.assert_equal(d.H[0, 2], tf.constant(0.00015966648064403503, dtype = tf.float64))
        tf.assert_equal(d.H[1, 1], tf.constant(9.424770000357663, dtype = tf.float64))
    
    # Tests if KE and PE values are added properly
    def test_add_H(self):
        d = data('potential_energy.dat', 3)
        d.kinetic_energy()
        d.potential_energy()
        tf.assert_equal(d.H[0, 0], tf.constant(9.424770000000002, dtype = tf.float64))
        # Sum of previous tests' H[1, 1]
        tf.assert_equal(d.H[1, 1], tf.constant(4.712388560987876 + 9.424770000357663, dtype = tf.float64))

    # Tests if minimum energy is found
    def test_energy(self):
        d = data('potential_energy.dat', 3)
        d.kinetic_energy()
        d.potential_energy()
        d.compute_coefficients()
        tf.assert_equal(d.min_e, tf.constant(4.702957640910531, dtype = tf.float64))
        
    # Tests if correct eigenvector (coefficients) is (are) pulled
    def test_coefficients(self):
        d = data('potential_energy.dat', 3)
        d.kinetic_energy()
        d.potential_energy()
        d.compute_coefficients()
        tf.assert_equal(d.min_v[0], tf.constant(0.6791193459007923, dtype = tf.float64))
        tf.assert_equal(d.min_v[1], tf.constant(0.00033284060889309826, dtype = tf.float64))
        tf.assert_equal(d.min_v[2], tf.constant(0.7340277945966407, dtype = tf.float64))

if __name__ == '__main__':
    unittest.main()
