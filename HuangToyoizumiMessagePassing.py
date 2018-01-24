#####################################################################
# 
# This is an implementation of the message passing algorithm of 
# Huang and Toyoizumi.
# 
# See Physical Review E 91, 050101(R) for algorithm details
# arxiv version of paper: arXiv:1502.00186v3 [cond-mat.stat-mech]
#
# Author: David C. Hoyle
#
# Date: 2018/01/24
#
# Licence: CC-BY 4.0 https://creativecommons.org/licenses/by/4.0/
#
#####################################################################


import math
import numpy as np

def doMessagePassing( connectivity, h_vec, phi_vec, omega_matrix, nIter ):
    """ Calculate a mean-field estimate of the log-partition function
        using the message passing algorithm of Huang and Toyoizumi
        
    Args:
        connectivity: Matrix indicating which nodes are coupled visible and 
                      hidden nodes are coupled. The matrix has dimensions 
                      number of visible nodes x number of hidden nodes. 
                      A matrix element with value 1 indicated the nodes are coupled
        h_vec: Biases (external field) of the hidden nodes expressed in
               Ising model formalism.
        phi_vec: Biases (external field) of the visible nodes expressed in
                 Ising model formalism.
        omega_matrix: Visible to hidden node couplings, expressed in
                      Ising model formalism.
        nIter: The number of iterations to perform of the message passing
               equations
 
    Returns: log-partition function of the Ising Model equivalent of the RBM
    """
    
    # extract dimensions
    nVisible = len( phi_vec )
    nHidden = len( h_vec )    
    
    # initialize u, g, and m matrices
    u_matrix = np.zeros( (nHidden, nVisible) )
    g_matrix = np.zeros( (nHidden, nVisible) )
    m_matrix = np.zeros( (nVisible, nHidden ) )
    
    # iterate message passing
    for iter in range(nIter):
        for i in range( nVisible ):
            for a in range( nHidden ):
                mean_field = 0
                for b in range( nHidden ):
                    if (connectivity[i, b]==1) & (b != a) :
                        mean_field += u_matrix[b, i] 
                        
                m_matrix[i, a] = math.tanh( phi_vec[i] + mean_field )
                
                
        # update cavity field
        for b in range( nHidden ):
            for i in range( nVisible ):
                g_field = 0
                for j in range( nVisible ):
                    if (connectivity[j, b])==1 & (j != i) :
                        g_field += (omega_matrix[b,j]*m_matrix[j,b])
                        
                u_matrix[b,i] = 0.5 * (math.log( math.cosh(h_vec[b] + g_field + omega_matrix[b,i]) ))
                u_matrix[b,i] -= 0.5 * (math.log( math.cosh(h_vec[b] + g_field - omega_matrix[b,i]) ))
                g_matrix[b, i] = g_field
                
    # calculate free energy
    xi2_matrix = np.zeros( (nHidden, nVisible) )
    xi2_vec = np.zeros( nHidden )
    g_vec = np.zeros( nHidden )
    
    for b in range( nHidden ):
        for i in range( nVisible ):

            if connectivity[i, b]==1 :
                g_vec[b] += omega_matrix[b,i] * m_matrix[i,b]
                xi2_vec[b] += math.pow( omega_matrix[b,i], 2 ) * (1.0 - math.pow( m_matrix[i, b], 2) )    
            
            for j in range( nVisible ):
                    if (connectivity[j, b]==1) & ( j != i ) :
                            xi2_matrix[b, i] += (math.pow( omega_matrix[b,j], 2) * ( 1-math.pow( m_matrix[j,b], 2) ))
    
    
    logZ = 0
    for b in range( nHidden ):
        logZ += float( nVisible - 1 ) * ( math.log( 2.0 ) + 0.5*xi2_vec[b] + math.log( math.cosh( h_vec[b] + g_vec[b] ) ) )
    
    
    for i in range( nVisible ):
        logZ -= phi_vec[i]
        
        xtmp1 = -2.0*phi_vec[i]
        for b in range( nHidden ):
            logZ -= math.log( 2.0 )
            logZ -= 0.5 * xi2_matrix[b, i]
            logZ -= math.log( math.cosh( h_vec[b] + g_matrix[b, i] + omega_matrix[b, i] ) )
            
            xtmp1 += math.log( math.cosh( h_vec[b] + g_matrix[b, i] - omega_matrix[b, i] ) )
            xtmp1 -= math.log( math.cosh( h_vec[b] + g_matrix[b, i] + omega_matrix[b, i] ) )
            
        logZ -= math.log( 1 + math.exp( xtmp1 ) )
        

    return logZ
