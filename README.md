# BONDSOAP

Create SOAP descriptors for atomic bonds in molecules

## Setup
```
python -m venv venv
pip install -r requirements.txt
```

## Quick example

- 1. Copy from https://www.kaggle.com/c/champs-scalar-coupling/data files
  - train.csv
  - structures.csv
  
    To folder 'data' in this directory

- 2. Run 
``` 
python makesoap.py --bonds ../champs_scalar_coupling/train.csv --structures ../champs_scalar_coupling/structures.csv --output data/ --cutoff 5 --nmax 4 --lmax 4 --id2 1 --njobs 4 --batch 10
```

To create SOAP descriptors for first 99 bonds in 'train.csv' into a CSV-file, in form (99, 1051), where the last label of the second dimension is the scalar coupling constant of the bond.
    
   
## About 

All atomic bonds of molecules, organized as in https://www.kaggle.com/c/champs-scalar-coupling/data files 'train/test.csv' and 'structures.csv', can be written as SOAP descriptors with or without labels (specified in the last column of 'train.csv') for machine learning tasks. For each bond, we define 'fingerprint points', such as 0.25,0.5 and 0.75, so that if the atoms in a bond are situated in the x-axis point 0 and 10, the SOAP descriptors would be evaluated at 2.5, 5, and 7.5.

The descriptor shape depends on the amount of complicacy imposed. This program uses the tqdm-library's SOAP-routines as described in https://singroup.github.io/dscribe/1.0.x/tutorials/descriptors/soap.html .
