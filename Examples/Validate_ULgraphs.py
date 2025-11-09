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
#
# G. Zamora-López & R. Brasselet "Sizing complex networks", Commun Phys 2:144 (2019)
#

"""
This script is an example to use and validate the generation and analytic
estimation of the ultra-long boundaries for graphs. We will:

1. Generate ultra-long graphs,
2. Numerically compute their pathlength and efficiency,
3. Analytically estimate their pathlength and efficiency, and
4. Compare the results.

PathLims works as an stand-alone package. For simplicity, however, this example
require that the pyGAlib package is installed for the manipulation, analysis and
generation of (di)graphs. pyGAlib can be installed from the Python Package Index
using `pip`. In a terminal, simply type:
    $ python3 -m pip install galib

See further information in https://github.com/gorkazl/pyGAlib
"""

# Standard library imports
from timeit import default_timer as timer
# Third party imports
import matplotlib.pyplot as plt
import numpy as np
import galib, galib.metrics_numba
# Local imports
import pathlims
import pathlims.limits as lims
import pathlims.generators as gens


################################################################################
# 0) PREPARE FOR THE CALCULATIONS
# Basic properties of the graphs.
N = 10
Lmax = int( 0.5*N*(N-1) )
Llist = np.arange(Lmax+1)
nL = len(Llist)

# Print some feedback
print( f"\nNetwork size, N: {N}" )


# 1) DO THE CALCULATIONS -- CONNECTED GRAPHS
time1 = timer()
pathlen_num = np.zeros(nL, np.float64)
pathlen_th = np.zeros(nL, np.float64)
effic_num_con = np.zeros(nL, np.float64)
effic_th_con = np.zeros(nL, np.float64)
pathlen_num[:N-1] = np.inf
pathlen_th[:N-1] = np.inf

print( 'Calculating results for CONNECTED graphs ...' )
for L in range(N-1, nL):
    # Generate the connected ultra-long graph and compute distance / efficiency
    net = gens.ULgraph_Connected(N,L)
    dij = galib.metrics_numba.FloydWarshall_Numba(net)
    pathlen_num[L] = ( dij.sum() - dij.trace() ) / (2*Lmax)
    eij = 1./dij
    effic_num_con[L] = ( eij.sum() - eij.trace() ) / (2*Lmax)

    # Calculate the results analitically
    ## Note: Pathlen_ULdigraph() returns 'inf', for L < N-1.
    pathlen_th[L] = lims.Pathlen_ULgraph(N,L)
    effic_th_con[L] = lims.Effic_ULgraph(N,L, connected=True)


# 2) DO THE CALCULATIONS -- DISCONNECTED GRAPHS
# In this case, we provide the results only for given values of L such that
# L = 1/2 M(M-1) with M = 2, ..., N. Results for remaining values of L will be
# treated in the future.
Mlist = np.arange(2,N+1)
nM = len(Mlist)
LMlist = np.zeros(nM, np.uint64)
effic_num_disco = np.zeros(nM, np.float64)
effic_th_disco = np.zeros(nM, np.float64)

print( 'Calculating results for DISCONNECTED graphs ...' )
for counter,M in enumerate(Mlist):
    # Get number of edges in an M-complete graph of order M
    LM = int( 0.5*M*(M-1) )
    LMlist[counter] = LM

    # Generate the M-complete graph and compute its efficiency
    net = gens.ULgraph_Disconnected_Mcomplete(N,M)
    dij = galib.metrics_numba.FloydWarshall_Numba(net)
    eij = 1./dij
    effic_num_disco[counter] = ( eij.sum() - eij.trace() ) / (2*Lmax)

    # Calculate the results analitically
    effic_th_disco[counter] = lims.Effic_ULgraph(N,LM, connected=False)

time2 = timer()
print(time2 - time1, 'seconds')


# 3) SUMMARISE THE RESULTS
print( '\nSUMMARY' )
diff_pathlen = abs(pathlen_num[N:] - pathlen_th[N:])
error_pathlen = diff_pathlen.sum()
print('Pathlength. Total error:', error_pathlen )

diff_effic_con = abs(effic_num_con - effic_th_con)
error_effic_con = diff_effic_con.sum()
print('Efficiency (connected). Total error:', error_effic_con )

diff_effic_disco = abs(effic_num_disco - effic_th_disco)
error_effic_disco = diff_effic_disco.sum()
print('Efficiency (disconnected). Total error:', error_effic_disco )


# 4) PLOT THE RESULTS
# 4.1) Plot comparison for pathlength
plt.figure()
plt.scatter(Llist, pathlen_num, s=10, color='#1f77b4', label='Numerical')
plt.plot(Llist, pathlen_th, color='gray', zorder=0, label='Theoretical')
plt.xlabel('Number of edges', fontsize=14)
plt.ylabel('Average pathlength', fontsize=14)
plt.xlim(-1,nL+1)
# plt.ylim(1,maxvalue)
plt.grid(ls='dotted')

plt.legend(loc='upper center', fontsize=10, scatterpoints=1, frameon=False)

# 4.2) Plot comparison for efficiency
plt.figure()
plt.scatter(Llist[N-1:], effic_num_con[N-1:], s=10, color='#1f77b4', label='Numerical (connected)', zorder=20)
plt.plot(Llist[N-1:], effic_th_con[N-1:], color='gray', label='Theoretical (connected)', zorder=10)
plt.scatter(LMlist, effic_num_disco, s=10, color='#d62728', label='Numerical (disconnected)', zorder=20)
plt.plot(LMlist, effic_th_disco, '--', color='gray', label='Theoretical (disconnected)', zorder=10)
plt.xlabel('Number of edges', fontsize=14)
plt.ylabel('Efficiency', fontsize=14)
plt.xlim(-1,nL+1)
plt.ylim(-0.1,1.1)
plt.grid(ls='dotted')

plt.legend(loc='upper left', fontsize=10, scatterpoints=1, frameon=False)


plt.show()



#
