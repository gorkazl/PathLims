# -*- coding: utf-8 -*-
#
# Copyright (c) 2018, Gorka Zamora-López and Romain Brasselet
# <gorka@Zamora-Lopez.xyz>
#
# Released under the Apache License, Version 2.0 (the "License");
# you may not use this software except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# REFERENCE AND CITATION
# When using PathLims, please cite:
# G. Zamora-López & R. Brasselet "Sizing complex networks", Commun Phys 2:144 (2019)
#

"""
This script is an example to use the ultra-short and the ultra-long boundaries
to study and evaluate the length (or efficiency) of empirical networks.
Case for directed networks. We will:

1. Read an empirical dataset.
2. Compute its average pathlength and efficiency,
3. Estimate the corresponding boundaries for pathlength and efficiency,
4. Calculate pathlength and efficiency of equivalent random graphs and lattices.
5. Compare the results.

PathLims works as an stand-alone package. For simplicity, however, this example
require that the pyGAlib package is installed for the manipulation, analysis and
generation of (di)graphs. pyGAlib can be installed from the Python Package Index
using `pip`. In a terminal, simply type:
    $ pip install galib

See further information in https://github.com/gorkazl/pyGAlib
"""

# Standard library imports
from timeit import default_timer as timer
# Third party imports
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import galib, galib.tools, galib.models, galib.metrics_numba
# Local imports
import pathlims.limits as lims
import pathlims.generators as gens


################################################################################
# 0) LOAD THE DATA AND PREPARE FOR CALCULATIONS
# Choose one network between (Dir_Cat.net, Dir_Elegans.net, Dir_Prison.net)
dataroot = 'Data/'
netfname = 'Dir_Cat.net'
net = galib.tools.LoadFromPajek(dataroot + netfname, getlabels=False)
# Binarise the network
net = np.where(net,1,0).astype(np.uint8)
N = len(net)

assert galib.Reciprocity(net) < 1.0, \
    "Network is Undirected. Use 'EmpiricalNets_Undirected.py' instead."

L = net.sum()
Lmax = int( N*(N-1) )

# Print some feedback
print('\nNetwork:', netfname)
print('N: %d\tL: %d\tDensity: %1.4f' %(N,L, L/Lmax) )


# 1) NUMERICALLY COMPUTE THE PATHLENGTH AND EFFICIENCY OF THE NETWORK
# Calculate the pairwise distance matrix and calculate average
dij = galib.metrics_numba.FloydWarshall_Numba(net)
dijsum = dij.sum()
if np.isinf(dijsum):
    pathlen_emp = np.inf
else:
    pathlen_emp = float( dijsum - dij.trace() ) / Lmax
# Calculate efficiency matrix and the average
eij = 1./dij
effic_emp = float( eij.sum() - eij.trace() ) / Lmax

# Check if network is connected
if dij.max() < N: connected = True
else: connected = False
print('Is the digraph strongly connected?', connected)


# 2) ESTIMATE THE BOUNDARIES FOR PATHLENGTH AND EFFICIENCY
# 2.1) ultra-short and ultra-long pathlength
pathlen_us = lims.Pathlen_USdigraph(N,L)
## NOTE: UL digraph model for L < (N-1) + 1/2 N(N-1) not known for all
## values of L. Pathlen_ULdigraph() returns an approximation in those cases.
pathlen_ul = lims.Pathlen_ULdigraph(N,L)

# 2.2) ultra-short efficiency
effic_us = lims.Effic_USdigraph(N,L)

# 2.3) ultra-long efficiency for connected digraphs
## NOTE: So far, no exact equation for efficiency of strongly connected
## ultra-long digraphs is known. We will generate two M-BS digraphs of
## order M and (M+1), we will calculate their efficiencies numerically
## and then interpolate the result
# Calculate the corresponding M value, given N and L
M = int( 0.5*(1.0 + np.sqrt(1.0 + 8.0*(L-N))) )
# Generate the UL digraph containing an M-BS. Calculate its efficiency
ulnet1 = gens.ULdigraph_Connected_Range1_MBS(N,M)
dij1 = galib.metrics_numba.FloydWarshall_Numba(ulnet1)
eij1 = 1./dij1
effic1 = ( eij1.sum() - eij1.trace() ) / Lmax
# Generate the UL digraph containing a (M+1)-BS. Calculate its efficiency
ulnet2 = gens.ULdigraph_Connected_Range1_MBS(N,M+1)
dij2 = galib.metrics_numba.FloydWarshall_Numba(ulnet2)
eij2 = 1./dij2
effic2 = ( eij2.sum() - eij2.trace() ) / Lmax
# Interpolate the result
LM1 = N + 0.5*M*(M-1)
LM2 = N + 0.5*(M+1)*M
effic_ul_con = effic1 + (L - LM1) * (effic2-effic1) / (LM2 - LM1)

# 2.4) ultra-long efficiency for disconnected digraphs
effic_ul_disco = lims.Effic_ULdigraph_Disconnected(N,L)


# 3) CALCULATE AVERAGE PATHLENGTH FOR EQUIVALENT RANDOM GRAPHS AND RING LATTICES
# 3.1) Equivalent random graphs, from ensemble
nrealiz = 100
print( "\nCalculating %d random realizations ..." %nrealiz)

pathlenlist = np.zeros(nrealiz, np.float)
efficlist = np.zeros(nrealiz, np.float)
for re in range(nrealiz):
    if re in np.arange(10,110,10):
        print( re )

    # Generate a random graph
    randnet = galib.models.RandomGraph(N,L, directed=True)
    # Calculate distance matrix and pathlength
    rdij = galib.metrics_numba.FloydWarshall_Numba(randnet)
    rdijsum = rdij.sum()
    if np.isinf(rdijsum):
        pathlenlist[re] = np.inf
    else:
        pathlenlist[re] = rdijsum - rdij.trace()
    # Calculate efficiency matrix and efficiency
    reij = 1./rdij
    efficlist[re] = reij.sum() - reij.trace()
print( 'Finished.' )

# Normalise the results
pathlenlist /= Lmax
efficlist /= Lmax

# Calculate the ensemble average pathlength. Ignore disconnected random graphs
goodidx = np.where(pathlenlist < np.inf)[0]
if goodidx.size > 0:
    pathlen_rand = pathlenlist[goodidx].mean()
else:
    pathlen_rand = np.nan
nbad = nrealiz - goodidx.size
if nbad:
    print( '%d disconnected random graph(s) found out of %d realizations' %(nbad,nrealiz) )
    print( 'Ignored from ensemble average pathlength' )
# Calculate the ensemble average pathlength
effic_rand = efficlist.mean()

# 3.2) Equivalent 1D ring lattices
# Generate a ring lattice of same number of edges
Lund = int(L/2)
latt = galib.models.Lattice1D_FixLinks(N,Lund)
# Calculate its distance matrix and the average pathlength
ldij = galib.metrics_numba.FloydWarshall_Numba(latt)
pathlen_latt = ( ldij.sum() - ldij.trace() ) / Lmax
# Calculate the efficiency matrix and the average
leij = 1. / ldij
effic_latt = ( leij.sum() - leij.trace() ) / Lmax


# 3) PLOT THE RESULTS
# 3.1) Print some feedback
print( '\nRESULTS --------------------------------------' )
print('\t\tPathlength\tEfficiency')
print('Ultra-short\t%2.4f\t\t%2.4f' %(pathlen_us, effic_us) )
print('Random\t\t%2.4f\t\t%2.4f' %(pathlen_rand, effic_rand) )
print('Empirical\t%2.4f\t\t%2.4f' %(pathlen_emp, effic_emp) )
print('Lattice\t\t%2.4f\t\t%2.4f' %(pathlen_latt, effic_latt) )
print('Ultra-long\t%2.4f\t\t%2.4f' %(pathlen_ul, effic_ul_con) )
print('UL discon.\t%2.4f\t\t%2.4f' %(np.inf, effic_ul_disco) )

# 3.2) Plot the figures
plt.figure()

# Plot the PATHLENGTH
plt.subplot(1,2,1)
plt.title('Pathlength', fontsize=14)
plt.axhline(y=1, linewidth=1.0, color='gray')
# Plot the boundaries
plt.axhline(y=pathlen_us, linewidth=1.0, color='gray', label='Ultra-short')
plt.axhspan(1, pathlen_us, facecolor='gray', alpha=0.3)
plt.axhline(y=pathlen_ul, linewidth=1.0, color='gold', label='Ultra-long')
plt.axhspan(pathlen_ul, pathlen_ul+5, facecolor='gold', alpha=0.3)
# Plot the random and lattice
plt.scatter(1,pathlen_rand, marker='_', s=500, label='Random')
plt.scatter(1,pathlen_latt, marker='_', s=500, label='Ring')
# Plot the empirical data
plt.scatter(1,pathlen_emp, marker='+', s=50, label='Real net')
plt.xticks(())
plt.ylim(1,pathlen_ul+1)

# Plot the EFFICIENCY
plt.subplot(1,2,2)
plt.title('Efficiency', fontsize=14)
# Plot the boundaries
plt.axhline(y=effic_us, linewidth=1.0, color='gray', label='Ultra-short')
plt.axhspan(effic_us, 1, facecolor='gray', alpha=0.3)
plt.axhline(y=effic_ul_con, linewidth=1.0, color='gold', label='Ultra-long')
plt.axhline(y=effic_ul_disco, ls='dashed', linewidth=1.0, color='gold', label='UL disco.')
plt.axhspan(0, effic_ul_disco, facecolor='gold', alpha=0.3)
# Plot the random and lattice
plt.scatter(1,effic_rand, marker='_', s=500, label='Random')
plt.scatter(1,effic_latt, marker='_', s=500, label='Ring')
# Plot the empirical data
plt.scatter(1,effic_emp, marker='+', s=50, label='Real net')
plt.xticks(())
plt.ylim(0,1)

plt.legend(loc='upper right', fontsize=9)

plt.tight_layout()


plt.show()



##
