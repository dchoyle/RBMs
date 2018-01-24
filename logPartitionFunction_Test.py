###################################################################
#
# Code to test the implementations of the Salakhutdinov & Murray
# AIS algorithm and the Huang and Toyoizumi message passing 
# algorithms. Both algorithms provide estimates of the 
# log-partition function of an RBM, where both the visible and
# hidden nodes have binary states.
# 
# Author: David C. Hoyle
#
# Date: 2018/01/24
#
# Licence: CC-BY 4.0 https://creativecommons.org/licenses/by/4.0/
#
#####################################################################


import imp
import numpy as np
import math
import time


# import the implementations of the AIS and message passing algorithms
rbmAIS = imp.load_source('module.name', './RBM_AIS.py')
rbmMP = imp.load_source('module.name', './HuangToyoizumiMessagePassing.py')


#### Set RBM parameter distribution characteristics
muA = 0.0 # mean of biases for the visible nodes
muB = 0.0 # mean of the biases for the hidden nodes
muJ_tilde = 0.0 # mean of the couplings between the visible and hidden nodes
sigma2A = 0.05 # variance of the biases for the visible nodes
sigma2B = 0.05 # variance of the biases for the hidden nodes
sigma2J_tilde = 1.0 # variance of the couplings between the visible and hidden nodes


### Do test on single RBM
nHidden = 4
nVisible = 20

# sample the biases and couplings (in Ising model formalism)
vbA_Ising = np.random.normal(loc=muA, scale=math.sqrt(sigma2A), size=nVisible )
hbA_Ising = np.random.normal( loc=muB, scale=math.sqrt(sigma2B), size=nHidden )
jMat_Ising = np.random.normal(loc=(muJ_tilde/nVisible), scale=math.sqrt(sigma2J_tilde/nVisible), size=(nVisible, nHidden))

# convert biases and coupling from Ising model formalism to Lattice Gas formalism
jMat_LatticeGas = 4.0 * jMat_Ising
vbA_LatticeGas = 2.0*(vbA_Ising - np.inner( jMat_Ising, np.ones(nHidden) ))
hbA_LatticeGas = 2.0*(hbA_Ising - np.inner( np.transpose(jMat_Ising), np.ones(nVisible) ))


## First calculate the exact value
exact_startTime = time.time()
freeEnergy_exact = -rbmAIS.exactPartitionFunction( vbA_LatticeGas, hbA_LatticeGas, np.transpose(jMat_LatticeGas))
exact_endTime = time.time()
exact_runTime = exact_endTime - exact_startTime    

## Now do the AIS calculation
nM = 100 # number of AIS runs

# create sequence of inverse temperatures. We will have as many value in the fifth of the transition
# to the B RBM, so that we have smaller steps in inverse temperture as we get closer to the fully coupled RBM.
# We could do this more elegantly, but the simple change in step size at beta=0.8 should suffice for 
# our testing purposes. 
nT = 10000
betaSeq = np.concatenate( [np.arange(0, 0.8, 0.0001), np.arange(0.8, (1.0 + (0.5/float(nT))), 1.0/float(nT)) ] )
ais_startTime = time.time()
freeEnergy_AIS = rbmAIS.calcRBMLogPartition( vbA_LatticeGas, vbA_LatticeGas, hbA_LatticeGas, hbA_LatticeGas, jMat_LatticeGas, betaSeq, nM, verbose=True )
ais_endTime = time.time()
ais_runTime = ais_endTime - ais_startTime

# Test of Message Passing code
connectivity_tmp = np.ones( (nVisible, nHidden) )
baseline_LatticeGas = (0.5*np.mean(vbA_LatticeGas)*float(nVisible)) + (0.5*np.mean(hbA_LatticeGas)*float(nHidden)) + (0.25*np.mean( jMat_LatticeGas ) * float(nHidden) * float(nVisible) )
mp_startTime = time.time()
freeEnergy_MP = -baseline_LatticeGas + rbmMP.doMessagePassing( connectivity_tmp, hbA_Ising, vbA_Ising, np.transpose(jMat_Ising), 20 )
mp_endTime = time.time()
mp_runTime = mp_endTime - mp_startTime


print( "Exact calc: logZ = " + str( freeEnergy_exact ) + "\t" + "Run time = " + str( exact_runTime ) + "\n" )
print( " " )
print( "AIS calc: logZ = " + str( -freeEnergy_AIS[0] ) + "\t" + "Run time = " + str( ais_runTime ) + "\n" )
print( " " )
print( "MP calc: logZ = " + str( freeEnergy_MP ) + "\t" + "Run time = " + str( mp_runTime ) + "\n" )
print( " " )



