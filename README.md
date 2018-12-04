[![Coverage Status](https://coveralls.io/repos/github/wfunkenbusch/1D_Schrodinger/badge.svg?branch=master)](https://coveralls.io/github/wfunkenbusch/1D_Schrodinger?branch=master)

Approximates the minimum energy and corresponding wavefunction of a quantum particle for an arbitrary potential field by numerically solving the one-dimensional time-independent Schrodinger equation using a Fourier series basis.

Implementation:
1. Needed outside modules: tensorflow, math, numpy, argparse, unittest (if unit testing)
2. Clone this repository using the following command: git clone https://github.com/wfunkenbusch/1D_Schrodinger.git 
3. Run the code using the following command:

Arguments:

* --FileName
    * type: string
    * default: 'potential_energy.dat'
    * The potential energy data file name given as x potential energy in a .dat file.

* --basis_size
    * type: integer
    * default: 5
    * The size of the Fourier basis set. The basis is {1, sin(x), cos(x), sin(2x), cos(2x), ...}.

* --c
    * type: float
    * default: 1.0
    * Scaling factor for the kinetic energy term in the Hamiltonian. Must be positive.

* --domain
    * type: list
    * default: [0, 9.42477]
    * The domain for the kinetic energy of the particle, given as a list containing the minimum and maximum bounds of the domain. Note that for physical results, the domain of the potential energy data should be contained within the domain.

Prints:

* Minimum Energy Level
    * An approximation of the minimum energy of the particle in the potential energy field. Note that due to the variational principle, this value will always be greater than the actual minimum energy of the particle.

* Coefficients for Fourier Basis
    * The coefficients for the wavefunction approximation, given in the same order as the basis. The wavefunction is:

$$\Psi = \sum_{i = 0}^{basis\_size}a_i F_i(x)$$

where $a_i$ is the coefficient for the $i$th basis element, and $F_i$ is the $i$th basis element. See --basis_size in Arguments for the basis.