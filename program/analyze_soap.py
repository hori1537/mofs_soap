from dscribe.descriptors import SOAP
import ase.io
import glob
import sys
import pandas as pd
import numpy as np

path_ = glob.glob('/home/garaken/mofs/data_20/*')
print(path_)

df_ = pd.DataFrame(columns = ['cif_file', 'soap'])
df_ = pd.DataFrame()
df_out = pd.DataFrame()

var_df = pd.DataFrame()
    
for cif_path in path_:

    structure = ase.io.read(cif_path)
    pos = [[0, 0, 0]]
    desc = SOAP(
        species=["H", "O", "Co", "C", "N", "Cd", "Na", "W", "P", "Cu", "Co", "Mn", "Cl", "Cd", "Tb", "S", "U", "La", "In", "Ce", "Pr", "Nd", "Sm", "Eu"],
        rcut=4.0,
        nmax=6,
        lmax=6,
        rbf="gto",
        sigma=0.4,
        sparse=False,
        periodic=False
    )
    out = desc.create(structure, positions=pos)

    print(cif_path)
    print(out)

    for value in out:
        vals = []
        for v in value:
            vals.append( v )


    tmp_se = pd.Series( vals )
    var_df = var_df.append( tmp_se, ignore_index=True )

print(var_df)

var_df = var_df.replace(0, np.nan)
print(var_df)

var_df = var_df.dropna(thresh = 5, axis=1)
print(var_df)

var_df = var_df.replace(np.nan, 0)
print(var_df)

var_df.to_csv('cif_soap_dropna.csv')