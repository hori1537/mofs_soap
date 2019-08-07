

import soaplite
import ase.io

structure = ase.io.read("test.cif")
pos = [[0, 0, 0]]
rcut = 4
nmax = 6
alphas, betas = soaplite.genBasis.getBasisFunc(rcut, nmax)
out = soaplite.get_soap_locals(
    structure,
    pos,
    alp=alphas,
    bet=betas,
    rCut=rcut,
    nMax=nmax,
    Lmax=6,
    eta=1/(2*0.4**2),
    all_atomtypes=[1]
)

print(out)


from dscribe.descriptors import SOAP
import ase.io

structure = ase.io.read("test.cif")
pos = [[0, 0, 0]]

desc = SOAP(
   species=["H"],
   rcut=4.0,
   nmax=6,
   lmax=6,
   rbf="gto",
   sigma=0.4,
   sparse=False,
   periodic=False
)
out = desc.create(structure, positions=pos)

print('out dscribe')
print(out)

