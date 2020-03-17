from symtensor.sym import random, einsum
from symtensor.tools.la import symsvd
from rsvd_symten import *
import time
import numpy as np
from sys import argv
import matplotlib as mpl
import matplotlib.pyplot as plt

# Set up Initial Tensor -----------------------------------------
# Parameters 
D = 12
chi = 110
Zn = 2
k = int(argv[2])
# Make a structured symtensor
fname = argv[1]
# Create a symtensor
ten1 = random((int(chi/Zn),int(D/Zn),int(D/Zn),int(chi/Zn),int(D/Zn)),
              ['+-++-',
               [range(Zn),range(Zn),range(Zn),range(Zn),range(Zn)],
               None,Zn],
              backend='numpy')
# put image into symtensor
A = np.array(mpl.image.imread(fname))
shape = A.shape
Aog = A.copy()
Aog2= A.copy()
Aog3= A.copy()
A = np.reshape(A,(-1))
nelem = np.prod(ten1.array.shape)
A = A[:nelem]
ten1.array = np.reshape(A,ten1.array.shape)
ten1.enforce_sym()

# Use existing SVD w/o trunc -----------------------------------------------
U,S,V = symsvd(ten1,[[0,1,2],[3,4]])
A2 = einsum('ijkl,lmn->ijkmn',U,einsum('jk,klm->jlm',S,V))
Aog = np.reshape(Aog,(-1))
Aog[:nelem] = np.reshape(A2.array,(-1))
Aog = np.reshape(Aog,shape)
f = plt.figure()
plt.imshow(Aog)

# Use existing SVD w/ trunc -----------------------------------------------
U,S,V = symsvd(ten1,[[0,1,2],[3,4]],truncate_mbd=k)
A2 = einsum('ijkl,lmn->ijkmn',U,einsum('jk,klm->jlm',S,V))
Aog2 = np.reshape(Aog2,(-1))
Aog2[:nelem] = np.reshape(A2.array,(-1))
Aog2 = np.reshape(Aog2,shape)
f = plt.figure()
plt.imshow(Aog2)

# Use symrandqr w/ trunc -----------------------------------------------------
Q,R = rsvd_symten(ten1,3,k,q=1,nover=k)
A3 = einsum('ijkl,lmn->ijkmn',Q,R)
Aog3 = np.reshape(Aog3,(-1))
Aog3[:nelem] = np.reshape(A3.array,(-1))
Aog3 = np.reshape(Aog3,shape)
f = plt.figure()
plt.imshow(Aog3)

plt.show()
