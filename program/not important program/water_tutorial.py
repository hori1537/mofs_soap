from dscribe.descriptors import SOAP

species = ["H", "C", "O", "N"]
rcut = 6.0
nmax = 8
lmax = 6

# Setting up the SOAP descriptor
soap = SOAP(
    species=species,
    periodic=False,
    rcut=rcut,
    nmax=nmax,
    lmax=lmax,
)

from ase.build import molecule

# Molecule created as an ASE.Atoms
water = molecule("H2O")

# Create SOAP output for the system
soap_water = soap.create(water, positions=[0])



print('soap_water')
print('soap_water.shape')
print(soap_water)
print(soap_water.shape)

# Create output for multiple system
samples = [molecule("H2O"), molecule("NO2"), molecule("CO2")]
positions = [[0], [1, 2], [1, 2]]
coulomb_matrices = soap.create(samples, positions)            # Serial
coulomb_matrices = soap.create(samples, positions, n_jobs=2)  # Parallel

from ase.build import molecule

# Molecule created as an ASE.Atoms
water = molecule("H2O")

# Create SOAP output for the system
soap_water = soap.create(water, positions=[0])


print('soap_water')
print('soap_water.shape')

print(soap_water)
print(soap_water.shape)

# Lets change the SOAP setup and see how the number of features changes
small_soap = SOAP(species=species, rcut=rcut, nmax=2, lmax=0)
big_soap = SOAP(species=species, rcut=rcut, nmax=9, lmax=9)
n_feat1 = small_soap.get_number_of_features()
n_feat2 = big_soap.get_number_of_features()
print('small_soap.get_number_of_features()', 'big_soap.get_number_of_features()')

print(n_feat1, n_feat2)

from ase.build import bulk

copper = bulk('Cu', 'fcc', a=3.6, cubic=True)

print('copper.get_pbc')

print(copper.get_pbc())
periodic_soap = SOAP(
    species=[29],
    rcut=rcut,
    nmax=nmax,
    lmax=nmax,
    periodic=True,
    sparse=False
)

soap_copper = periodic_soap.create(copper)

print('soap_copper')

print(soap_copper)

print('soap_copper.sum')
print(soap_copper.sum(axis=1))


soap = SOAP(
    species=species,
    rcut=rcut,
    nmax=nmax,
    lmax=lmax,
    sparse=True
)
soap_water = soap.create(water)
print('type(soap_water)')
print(type(soap_water))

soap = SOAP(
    species=species,
    rcut=rcut,
    nmax=nmax,
    lmax=lmax,
    sparse=False
)
soap_water = soap.create(water)
print(type(soap_water))

average_soap = SOAP(
    species=species,
    rcut=rcut,
    nmax=nmax,
    lmax=lmax,
    average=True,
    sparse=False
)

soap_water = average_soap.create(water)
print("average soap water", soap_water.shape)

methanol = molecule('CH3OH')
soap_methanol = average_soap.create(methanol)
print("average soap methanol", soap_methanol.shape)

h2o2 = molecule('H2O2')

from scipy.spatial.distance import pdist, squareform
import numpy as np

import sys
sys.exit()

molecules = np.vstack([soap_water, soap_methanol, soap_peroxide])
distance = squareform(pdist(molecules))
print("distance matrix: water - methanol - H2O2")
print(distance)