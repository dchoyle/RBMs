#####################################################################
# 
# This is an implementation of the methodology outlined in 
# Salakhutdinov & Murray, "On the Quantitative Analysis of Deep Belief Networks" 
# in Proceedings of ICML2008.
#
# The methodology estimates the log of the partition function for an RBM 
# (the B RBM) which has full coupling between the visible and hidden nodes
# of the RBM. The log-partition function is estimated via a series of 
# annealed importance sampling runs, starting from an A RBM which has no 
# coupling between the visible and hidden nodes. The transition the
# Boltzman distribution of the A RBM to the Boltzman distribution of the
# B RBM is handled via a parameter, beta, which takes a sequence of values. 
# We call beta the inverse temperature as it is used to multiply the 
# Hamiltonian in the process of transitioning between the A RBM to the 
# B RBM. The states of the visible and hidden nodes are binary, 
# i.e. we can express the model in an Ising model formalism or a 
# Lattice Gas formalism (the two formalisms are isomorphic to each other).
#
# Author: David C. Hoyle
#
# Date: 2018/01/24
#
# Licence: CC-BY 4.0 https://creativecommons.org/licenses/by/4.0/
#
#####################################################################


import numpy as np
import math

def sigmoid(x):
    """Calculates the sigmoid of the input value.

    Args:
      Takes a float x.

    Returns:
      Returns the sigmoid function value for x.
    """

    sigmoid = 1 / (1 + np.exp(-x))
    
    return sigmoid
    
def sampleHiddenFromVisible( beta, hiddenBias, visible, W ):
    """Samples values of hidden nodes given values of the visible nodes.

    Args:
      beta: Inverse temperature.
      hiddenBias: Array of biases, one for each hidden node.
      visible: The values of the visible nodes.
      W: 2-dimensional array of the visible-to-hidden connection weights

    Returns: An array of sampled values for the hidden nodes
    """

    h = np.zeros(W.shape[1])
        
    linearPredictor = (1-beta)*(np.dot(visible, W) + hiddenBias)
    h = ( np.random.random(size=hiddenBias.shape[0]) < sigmoid(linearPredictor) ) * 1
    
    return h
    
def sampleFromBias( beta, bias ):
    """Samples states of nodes given their bias and the inverse temperature 

    Args:
      beta: Inverse temperature
      bias: The biases of the nodes

    Returns: An array of sampled values for the nodes
    """
    h = np.zeros( bias.shape[0] )
    
    h = ( np.random.random(size=bias.shape[0]) < sigmoid( beta*bias ) ) * 1 
    
    return h
    
def sampleVisibleFromHidden( beta, visBiasA, visBiasB, wB, hB ):
    """Samples values of visible nodes given parameter values
       for two RBMs, A and B.

    Args:
      beta: Inverse temperature.
      visBiasA: Array of biases, one for each visible node for the A RBM.
      visBiasB: Array of biases, one for each visible node for the B RBM.
      wB: 2-dimensional array of the visible-to-hidden connection weights
          for the B RBM.
      hB: The values of the hidden nodes of the B RBM.

    Returns: An array of sampled values for the visible nodes of the B RBM.
    """
    v = np.zeros(wB.shape[0])
     
    linearPredictor = (1-beta)*visBiasA
    linearPredictor = linearPredictor + (beta*(np.dot(wB, hB) + visBiasB))

    v = ( np.random.random(size=wB.shape[0]) < sigmoid(linearPredictor) ) * 1
     
    return v

def calcLogPStar( beta, visBiasA, visBiasB, hiddenBiasA, hiddenBiasB, wB, visible ):
    """Calculates the log of the (un-normalized) probability of the visible state, 
      for a distribution intermediate between that of the A and B RBM.

    Args:
      beta: The inverse temperature.
      visBiasA: Array of biases, one for each visible node for the A RBM.
      visBiasB: Array of biases, one for each visible node for the B RBM.
      hiddenBiasA: Array of biases, one for each hidden node for the A RBM.
      hiddenBiasB: Array of biases, one for each hidden node for the B RBM.
      wB: 2-dimensional array of the visible-to-hidden connection weights
          for the B RBM.
      visible: The visible state for which the log-probability is required.

    Returns: The log of the un-normalized probability p*(visible).
    """

    logPStar = ((1-beta)*np.inner(visBiasA, visible)) + (beta*np.inner(visBiasB, visible))
    
    logPStar = logPStar + np.sum(np.log( 1 + np.exp((1-beta)*hiddenBiasA)))
        
    logPStar = logPStar + np.sum(np.log( 1 + np.exp(beta*(np.dot(visible, wB) + hiddenBiasB)) ))
        
    return logPStar
        
def singleAISRun( visBiasA, visBiasB, hiddenBiasA, hiddenBiasB, wB, betaSeq ):
    """Performs a single Annealed Importance Sampling (AIS) run.

    Args:
      visBiasA: The biases of the visible nodes in the A RBM
      visBiasB: The biases of the visible nodes in the B RBM
      hiddenBiasA: The biases of the hidden nodes in the A RBM
      hiddenBiasB: The biases of the hidden nodes in the B RBM
      wB: The visible-to-hidden couplings in the B RBM
      betaSeq: The sequence of inverse temperatures to transition between
               A RBM and the B RBM.

    Returns: A single AIS run weight.
    """
    logWeight = 0.0
    
    vCurrent = sampleFromBias(1.0, visBiasA)    
    
    for i in range(1, len(betaSeq)):
        #print( i, "\t", logWeight )
        logWeight = logWeight + calcLogPStar( betaSeq[i], visBiasA, visBiasB, hiddenBiasA, hiddenBiasB, wB, vCurrent )
        logWeight = logWeight - calcLogPStar( betaSeq[i-1], visBiasA, visBiasB, hiddenBiasA, hiddenBiasB, wB, vCurrent )

        hiddenB_tmp = sampleHiddenFromVisible( 1-betaSeq[i], hiddenBiasB, vCurrent, wB )
        
        vCurrent = sampleVisibleFromHidden( betaSeq[i], visBiasA, visBiasB, wB, hiddenB_tmp )
        
    return logWeight
    
def calcRBMLogPartition( visBiasA, visBiasB, hiddenBiasA, hiddenBiasB, wB, betaSeq, nM, verbose=True ):
    """Calculate the log partition function for the B RBM.

    Args:
      visBiasA: The biases of the visible nodes in the A RBM
      visBiasB: The biases of the visible nodes in the B RBM
      hiddenBiasA: The biases of the hidden nodes in the A RBM
      hiddenBiasB: The biases of the hidden nodes in the B RBM
      wB: The visible-to-hidden couplings in the B RBM
      betaSeq: The sequence of inverse temperatures to transition between
               A RBM and the B RBM.


    Returns: An array containing two elements. 
             The first element is the log of the AIS estimate of the partition function.
             The second element is the log of the standard error of the AIS estimate of  
             the ratio of the B RBM partition function to the A RBM partition function.
    """

    # calculate log-partition function of the reference A RBM.
    logPartitionA = 0
    for i in range(hiddenBiasA.shape[0]):
        logPartitionA = logPartitionA + math.log( 1 + math.exp(hiddenBiasA[i]) )
    
    for i in range(visBiasA.shape[0]):
        logPartitionA = logPartitionA + math.log( 1 + math.exp(visBiasA[i]) )


    # initialize log of sum of AIS weights by performing the first AIS run
    logSumWt = singleAISRun( visBiasA, visBiasB, hiddenBiasA, hiddenBiasB, wB, betaSeq )
    logSumWt2 = 2.0 * logSumWt

    # perform the remaining AIS runs
    for i in range(nM):
        logWt = singleAISRun( visBiasA, visBiasB, hiddenBiasA, hiddenBiasB, wB, betaSeq )
        if verbose:
            print( "AIS run: ", i, "Weight = ", logWt )        

        deltaLog = logSumWt - logWt
        if deltaLog > 20:
            logSumWt = logSumWt
        elif deltaLog < -20:
            logSumWt = logWt
        else:
            logSumWt = logSumWt + math.log( 1 + math.exp(-deltaLog ) )
            
        deltaLog2 = logSumWt2 - (2.0*logWt)
        if deltaLog2 > 20:
            logSumWt2 = logSumWt2
        elif deltaLog2 < -20:
            logSumWt2 = 2.0*logWt
        else:
            logSumWt2 = logSumWt2 + math.log( 1 + math.exp(-deltaLog2 ) )
            
    logRatioVar = logSumWt2 - math.log( nM+1 ) + math.log( 1 - math.exp(2.0*logSumWt - logSumWt2 - math.log(nM+1)) ) 
    logRatioVar -= math.log( nM+1 )
    logPartition = logSumWt  - math.log( nM+1 ) + logPartitionA
    
    results = [logPartition, 0.5*logRatioVar]

    return results


## Code for evaluation of log partition function via exhaustive enumeration 
## of states. This should only be used for small scale systems.
    
def generateExhaustiveBitStrings( nBits ):
    """ Generates all the binary numbers representable in nBits

    Args:
      nBits: The length of the binary numbers = number of nodes 
             whose states we want to represent.

    Returns: An array of integers (values 0/1). Each row of the array 
             represents a state of the nodes.

    """
    
    nString = 2**nBits 
    intSet = range( nString )
    
    bitSet = np.zeros( [ nString, nBits ] )
    for s in range(nString):
        bitString = bin(intSet[s])[2:].zfill(nBits)
            
        bitStringList = list( bitString )
        for b in range(nBits):
            bitSet[s,b] = bitStringList[b]
    
    return bitSet
                    

def hamiltonian( v, h, q, k, J ):
    """Calculate the energy given the visible and hidden states, and the RBM parameters,

    Args:
      v: The visible node states.
      h: The hidden node states
      q: The visible node biases.
      k: The hidden node biases.
      J: The visible-to-hidden node couplings.

    Returns: The energy of the configuration.
    """
    energy = np.inner(v, q) + np.inner( h, k)
    energy = energy + np.inner( h, np.inner(J, v))

    return -energy
    
def exactPartitionFunction( q, k, J ):
    """Calculates the exact value of the log-partition function.

    Args:
      q: The visible node biases.
      k: The hidden node biases.
      J: The visible-to-hidden node couplings.

    Returns: The exact log-partition function. 
             Note the calculation cannot be exact due to being a numerical
             implementation. We have approximated the addition of Boltzman 
             weights to avoid underflow/overflow errors.
    """

    # extract the number of visible and hidden nodes
    nHidden = len( k )
    nVisible = len( q )
    
    # generate all the possible states for the hidden nodes 
    # and all the possible states for the visible visible nodes 
    hiddenStateArray = generateExhaustiveBitStrings( nHidden )
    visibleStateArray = generateExhaustiveBitStrings( nVisible )

    # initialize the log-partition function
    logZ = 0.0

    # loop over the hidden and visible state arrays
    for hIdx in range(len(hiddenStateArray)):
        for vIdx in range(len(visibleStateArray)):
    
            visibleState = visibleStateArray[vIdx,]
            hiddenState = hiddenStateArray[hIdx,]
    
            # calculate the energy for the current configuration
            energy = hamiltonian( visibleState, hiddenState, q, k, J)    
            
            # update the log-partition function
            deltaLog = logZ + energy
            if deltaLog > 30.0:
                logZ = logZ
            elif deltaLog < -30.0:
                logZ = -energy
            else:
                logZ = logZ + math.log( 1 + math.exp(-deltaLog ) )
            
    return logZ 

