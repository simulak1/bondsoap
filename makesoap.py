import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ase import Atoms
#from IPython.core.display import display, HTML
from seaborn import countplot,distplot
import sys
from dscribe.descriptors import SOAP

from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score as R2
from scipy.special import comb


import random
from tqdm import tqdm
import time
import argparse

def get_args():

    args_parser = argparse.ArgumentParser()

    args_parser.add_argument(
	'--bonds',
        help='''
        The csv-file containing the requested bonds of molecules specified with 
        keyword "structures". Columns must be organized as
        
        id, molecule_name, atom_index_0, atom_index_1, type, label,

        Above, the label is required when keyword purpose=train, and when given, 
        it gives the target value of the bond.
        
        ''',
	required=True
    )

    args_parser.add_argument(
        '--structures',
        help='''
        The csv-file containing the structures of the molecules, with columns as
        
        molecule_name, atom_index, atom_name, x, y, z
        
        ''',
        required=True
    )
    
    args_parser.add_argument(
	'--output',
        help='Path to the output folder.',
	required=True
    )
    
    # Data files arguments
    args_parser.add_argument(
        '--cutoff',
        help='SOAP cutoff',
        type=int,
        required=True
    )

    # Data files arguments
    args_parser.add_argument(
        '--nmax',
        help='SOAP nmax',
        type=int,
        required=True
    )

    # Data files arguments
    args_parser.add_argument(
        '--lmax',
        help='SOAP lmax',
        type=int,
        required=True
    )

    args_parser.add_argument(
	'--njobs',
        help='Number of cores to use for soap conversion.',
	type=int,
	default=1
    )

    args_parser.add_argument(
	'--batch',
        help='Number of descriptors to include in one file.',
	type=int,
	default=1
    )
    
    args_parser.add_argument(
        '--id1',
        help='Start making SOAP descriptors from bond index id1*batch.',
        type=int,
        default=0
    )

    args_parser.add_argument(
        '--id2',
        help='Make (id2-id1)*batch batches, i.e. separate files with batch bond descriptors.',
        type=int,
        default=1,
        required=True
    )

    args_parser.add_argument(
        '--purpose',
        help='Purpose of the data: test or train?',
        default='train'
    )

    args_parser.add_argument(
        '--fingerprints',
        help='''
        Specify the partial distances in which you want to calculate
        a SOAP descriptor between the atoms.
        ''',
        nargs='+',
        type=float,
        default=[.25,.5,.75]
    )
    
    args_parser.add_argument(
        '--average-points',
        help='Average the obtained SOAP arrays over the fingerpint points.',
        default='yes'
    )


    return args_parser.parse_args()
                    


def makeSOAP(data,geom_data,ids,partials,cutoff,nmax,lmax,ifil,train,average,output,njobs):
    '''
    85003 molecules in train data.
    45772 molecules in test data.

    '''

    molecules=data['molecule_name'].unique()[ids]
            
    soaper = SOAP(
        rcut=cutoff,
        nmax=nmax,
        lmax=lmax,
        species=geom_data['atom'].unique(),
        sparse=False
    )

    soaps=[]
    moli=0
    for molecule in tqdm(molecules):
        coords = geom_data[['x','y','z']][(geom_data['molecule_name']==molecule)].values
        symbols = geom_data['atom'][(geom_data['molecule_name']==molecule)].values
        indices = data[['atom_index_0','atom_index_1']][(data['molecule_name']==molecule)].values

        if train and moli==0:
            labels = data['scalar_coupling_constant'][(data['molecule_name']==molecule)].values
        elif train and moli>0:
            labels_temp = data['scalar_coupling_constant'][(data['molecule_name']==molecule)].values
            labels=np.concatenate((labels,labels_temp))
        moli=moli+1
            
        N=indices.shape[0]
        for i in range(N):
            vec1 = coords[indices[i,0]]
            vec2 = coords[indices[i,1]]
            pos = []
            for a in partials:
                point=vec1+a*(vec2-vec1)
                pos.append([point[0],point[1],point[2]])
                
            soaps.append(soaper.create(Atoms(symbols,positions=coords),positions=pos,n_jobs=njobs))
    if average:
        array_a = np.mean(np.array((soaps)),axis=1)
    else:
        
        array_a=np.array((soaps))

        print(array_a.shape)
                
        N=array_a.shape[0]
        array_a = array_a.reshape(N,-1)
        
        print(array_a.shape)

    
    if train:
        array_to_save=np.concatenate((array_a,labels.reshape(-1,1)),axis=1)
    else:
        array_to_save = array_a

    print("Saving soap data of shape ",array_to_save.shape)
        
    array_in_csv = pd.DataFrame(array_to_save)
    array_in_csv.to_csv(output+str(cutoff)+str(nmax)+str(lmax)+'_'+str(ifil)+'.csv',header=False,index=False)

def make_soap_files(args):
    print(args.fingerprints)

    cutoff = args.cutoff
    nmax   = args.nmax
    lmax   = args.lmax
    
    id1 = args.id1
    id2 = args.id2

    njobs=args.njobs
    
    average=False
    if args.average_points=='yes':
        average=True
    
    # Load data
    geom_data=pd.read_csv(args.structures)
    # get molecules
    bond_data=pd.read_csv(args.bonds)

    # Is there labels in bond data (train_soap=True)?
    train_soap=False
    if(args.purpose=='train'):
        train_soap=True
        
    for i in range(id1,id2):
        startid=i*args.batch;stopid=(i+1)*args.batch
        print('Startid: '+str(startid)+', stopid: '+str(stopid))
        makeSOAP(data=bond_data,
                 geom_data=geom_data,
                 ids=range(startid,stopid),
                 partials=args.fingerprints,
                 cutoff=cutoff,
                 nmax=nmax,
                 lmax=lmax,
                 ifil=i,
                 train=train_soap,
                 average=average,
                 output=args.output,
                 njobs=njobs)
        
                        

def main():
    '''
    Program to make soap files for DNN. Partials, meaning the
    point specification between bonds for SOAP, is not yet implemented.
    '''
    
    args = get_args()

    make_soap_files(args)

    
    
if __name__ == '__main__':
    main()
                                                    
