"""
This script is an example to use the ultra-short and the ultra-long boundaries
to study and evaluate the length (or efficiency) of empirical networks. We will:

1. Read an empirical dataset.
2. Compute its average pathlength and efficiency,
3. Estimate the corresponding boundaries for pathlength and efficiency,
4. Calculate pathlength and efficiency of equivalent random graphs and lattices.
5. Compare the results.
"""
from __future__ import division, print_function, absolute_import

__author__ = "Gorka Zamora-Lopez"
__email__ = "galib@zamora-lopez.xyz"
__copyright__ = "Copyright 2018"
__license__ = "GPL"
__update__="31/08/2018"

# Standard library imports
from timeit import default_timer as timer
# Third party imports
import matplotlib.pyplot as plt
import numpy as np
# Local imports
import pathbounds
from pathbounds.limits import (Pathlen_USgraph, Pathlen_ULgraph,
                            Effic_USgraph, Effic_ULgraph)
# from pathbounds.models import USgraph
from pathbounds.helpers import (LoadFromPajek, Reciprocity, FloydWarshall,
                                RandomGraph, Lattice1D_FixLinks)


################################################################################
# 0) LOAD THE DATA AND PREPARE FOR CALCULATIONS
# Load the network
dataroot = 'Data/'
netfname = 'Und_Zachary.net'
net = LoadFromPajek(dataroot + netfname, getlabels=False)
# Binarise the network
net = np.where(net,1,0).astype(np.uint8)
N = len(net)
assert Reciprocity(net) == 1.0, \
            "Network is directed. Use 'AnalyseDatasets_Directed.py' instead."

L = 0.5*net.sum()
Lmax = int( 0.5*N*(N-1) )
density = L / Lmax
# Print some feedback
print('Network:', netfname)
print('N: %d\tL: %d\tDensity: %1.4f' %(N,L, density) )


# 1) NUMERICALLY COMPUTE THE PATHLENGTH AND EFFICIENCY OF THE NETWORK
# Calculate the pairwise distance matrix and calculate average
dij = FloydWarshall(net)
pathlen_emp = ( dij.sum() - dij.trace() ) / (2*Lmax)
# Calculate efficiency matrix and the average
eij = 1./dij
effic_emp = ( eij.sum() - eij.trace() ) / (2*Lmax)

# Check if network is connected
if dij.max() < N: connected = True
else: connected = False
print('Graph is connected?', connected)


# 2) ESTIMATE THE BOUNDARIES FOR PATHLENGTH AND EFFICIENCY
# 2.1) ultra-short and ultra-long pathlength
pathlen_us = Pathlen_USgraph(N,L)
pathlen_ul = Pathlen_ULgraph(N,L)

# 2.2) ultra-short and ultra-long efficiency
effic_us = Effic_USgraph(N,L)
effic_ul = Effic_ULgraph(N,L, connected=connected)


# 3) CALCULATE AVERAGE PATHLENGTH FOR EQUIVALENT RANDOM GRAPHS AND RING LATTICES
# 3.1) Equivalent random graphs, from ensemble
nrealiz = 10
pathlenlist = np.zeros(nrealiz, np.float)
efficlist = np.zeros(nrealiz, np.float)
for re in xrange(nrealiz):
    # Generate a random graph
    randnet = RandomGraph(N,L, directed=False)
    # Calculate distance matrix and pathlength
    rdij = FloydWarshall(randnet)
    pathlenlist[re] = rdij.sum() - rdij.trace()
    # Calculate efficiency matrix and efficiency
    reij = 1./rdij
    efficlist[re] = reij.sum() - reij.trace()

# Normalise the results
pathlenlist /= 2*Lmax
efficlist /= 2*Lmax

# Calculate the ensemble average pathlength. Ignore disconnected random graphs
goodidx = np.where(pathlenlist < np.inf)[0]
pathlen_rand = pathlenlist[goodidx].mean()
nbad = nrealiz - len(goodidx)
if nbad:
    print( '%d disconnected random graph(s) found out of %d realizations' %(nbad,nrealiz) )
    print( 'Ignored from ensemble average pathlength' )
# Calculate the ensemble average pathlength
effic_rand = efficlist.mean()

# 3.2) Equivalent 1D ring lattices
# Generate a ring lattice of same number of edges
latt = Lattice1D_FixLinks(N,L)
# Calculate its distance matrix and the average pathlength
ldij = FloydWarshall(latt)
pathlen_latt = ( ldij.sum() - ldij.trace() ) / (2*Lmax)
# Calculate the efficiency matrix and the average
leij = 1. / ldij
effic_latt = ( leij.sum() - leij.trace() ) / (2*Lmax)


# 3) PLOT THE RESULTS
# 3.1) Print some feedback
print('\t\tPathlength\tEfficiency')
print('Ultra-short\t%2.4f\t\t%2.4f' %(pathlen_us, effic_us) )
print('Random\t\t%2.4f\t\t%2.4f' %(pathlen_rand, effic_rand) )
print('Empirical\t%2.4f\t\t%2.4f' %(pathlen_emp, effic_emp) )
print('Lattice\t\t%2.4f\t\t%2.4f' %(pathlen_latt, effic_latt) )
print('Ultra-long\t%2.4f\t\t%2.4f' %(pathlen_ul, effic_ul) )

# 3.2) Plot the results
plt.figure()
# Plot the PATHLENGTH
plt.subplot(1,2,1)
plt.title('Pathlength', fontsize=14)
plt.axhline(y=1, linewidth=1.0, color='gray')
# Plot the boundaries
plt.axhline(y=pathlen_us, linewidth=1.0, color='gray', label='Ultra-short')
plt.axhspan(0, pathlen_us, facecolor='gray', alpha=0.3)
plt.axhline(y=pathlen_ul, linewidth=1.0, color='gold', label='Ultra-long')
plt.axhspan(pathlen_ul, pathlen_ul+5, facecolor='gold', alpha=0.3)
# Plot the random and lattice
plt.scatter(1,pathlen_rand, marker='_', s=500, label='Random')
plt.scatter(1,pathlen_latt, marker='_', s=500, label='Ring')
# Plot the empirical data
plt.scatter(1,pathlen_emp, marker='+', s=50, label='Real net')
plt.xticks(())
plt.ylim(0,pathlen_ul+1)

# Plot the EFFICIENCY
plt.subplot(1,2,2)
plt.title('Efficiency', fontsize=14)
# Plot the boundaries
plt.axhline(y=effic_us, linewidth=1.0, color='gray', label='Ultra-short')
plt.axhspan(effic_us, 1, facecolor='gray', alpha=0.3)
plt.axhline(y=effic_ul, linewidth=1.0, color='gold', label='Ultra-long')
plt.axhline(y=density, ls='dashed', linewidth=1.0, color='gold', label='UL disco.')
plt.axhspan(0, effic_ul, facecolor='gold', alpha=0.3)
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
