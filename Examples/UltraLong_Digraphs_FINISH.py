"""
This script is an example to use and validate the generation and analytic
estimation of the ultra-long boundaries for directed graphs. We will:

1. Generate ultra-long directed graphs,
2. Numerically compute their pathlength and efficiency,
3. Analytically estimate their pathlength and efficiency, and
4. Compare the results.
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
from pathbounds.limits import (Pathlen_ULdigraph, Pathlen_ULdigraph_Range1_MBS,
                            Effic_dULdigraph )
from pathbounds.models import (ULdigraph_Connected_Range1_MBS,
                            ULdigraph_Connected_Range2, ULdigraph_Disconnected_Range1,
                            ULdigraph_Disconnected_Range2 )
from pathbounds.helpers import FloydWarshall

## WRITE THIS ONE ANOTHER DAY !!!

################################################################################
# 0) PREPARE FOR THE CALCULATIONS
# Basic properties of the graphs.
N = 10
Lmax = int( 0.5*N*(N-1) )
Llist = np.arange(Lmax+1)
nL = len(Llist)

# 1) DO THE CALCULATIONS -- CONNECTED GRAPHS
# 1.0) Prepare arrays
pathlen_num = np.zeros(nL, np.float)
pathlen_th = np.zeros(nL, np.float)
effic_num_con = np.zeros(nL, np.float)
effic_th_con = np.zeros(nL, np.float)
pathlen_num[:N-1] = np.inf
pathlen_th[:N-1] = np.inf

time1 = timer()
for L in xrange(N-1, nL):
    # 1.1) Generate the connected ultra-long graph and compute distance / efficiency
    net = ULgraph_Connected(N,L)
    dij = FloydWarshall(net)
    pathlen_num[L] = dij.sum() - dij.trace()
    eij = 1./dij
    effic_num_con[L] = eij.sum() - eij.trace()

    # 1.2) Calculate the results analitically
    ## Note: Pathlen_ULdigraph() returns 'inf', for L < N-1.
    # pathlen_th[L] = Pathlen_ULgraph(N,L)
    # effic_th_con[L] = Effic_ULgraph(N,L, connected=True)

# Normalise the numerical results
pathlen_num /= 2*Lmax
effic_num_con /= 2*Lmax


# 2) DO THE CALCULATIONS -- DISCONNECTED GRAPHS
# 2.0) Prepare arrays
Mlist = np.arange(2,N+1)
nM = len(Mlist)
LMlist = np.zeros(nM, np.uint)
effic_num_disco = np.zeros(nM, np.float)
effic_th_disco = np.zeros(nM, np.float)

for counter,M in enumerate(Mlist):
    # Get number of edges in an M-complete graph of order M
    LM = int( 0.5*M*(M-1) )
    LMlist[counter] = LM

    # 2.1) Generate the M-complete graph and compute its efficiency
    net = ULgraph_Disconnected_Mcomplete(N,M)
    dij = FloydWarshall(net)
    eij = 1./dij
    effic_num_disco[counter] = eij.sum() - eij.trace()

    # 1.2) Calculate the results analitically
    effic_th_disco[counter] = Effic_ULgraph(N,LM, connected=False)

# Normalise the numerical results
effic_num_disco /= 2*Lmax

time2 = timer()
print(time2 - time1, 'seconds')


# 3) SUMMARISE THE RESULTS
diff_pathlen = abs(pathlen_num[N:] - pathlen_th[N:])
error_pathlen = diff_pathlen.sum()
print('Pathlength. Total error:', error_pathlen )

diff_effic_con = abs(effic_num_con - effic_th_con)
error_effic_con = diff_effic_con.sum()
print('Efficiency. Total error:', error_effic_con )

diff_effic_disco = abs(effic_num_disco - effic_th_disco)
error_effic_disco = diff_effic_disco.sum()
print('Efficiency. Total error:', error_effic_disco )


# 3) PLOT THE RESULTS
# 3.1) Plot comparison for pathlength
plt.figure()
plt.title('Total error: %3.4f' %error_pathlen, fontsize=12)
plt.scatter(Llist, pathlen_num, s=10, color='#1f77b4', label='Numerical')
plt.plot(Llist, pathlen_th, color='gray', zorder=0, label='Theoretical')
plt.xlabel('Number of edges', fontsize=14)
plt.ylabel('Average pathlength', fontsize=14)
plt.xlim(-1,nL+1)
# plt.ylim(1,maxvalue)
plt.grid(ls='dotted')

plt.legend(loc='upper center', fontsize=10, scatterpoints=1, frameon=False)

# 3.2) Plot comparison for efficiency
plt.figure()
plt.title('Error (con): %3.4f  /  Error (disco): %3.4f' \
                %(error_effic_con, error_effic_disco), fontsize=12)
plt.scatter(Llist[N-1:], effic_num_con[N-1:], s=10, color='#d62728', label='Num. (Connected)', zorder=20)
plt.plot(Llist[N-1:], effic_th_con[N-1:], color='gray', label='Theoretical', zorder=10)
plt.scatter(LMlist, effic_num_disco, s=10, color='#2ca02c', label='Num. (Disconnected)', zorder=20)
plt.plot(LMlist, effic_th_disco, '--', color='gray', label='Theoretical', zorder=10)
plt.xlabel('Number of edges', fontsize=14)
plt.ylabel('Efficiency', fontsize=14)
plt.xlim(-1,nL+1)
plt.ylim(-0.1,1.1)
plt.grid(ls='dotted')

plt.legend(loc='upper left', fontsize=10, scatterpoints=1, frameon=False)


# # 3.3) Validation result for pathlength
# plt.figure()
# maxvalue = max( pathlen_num[N:].max(), pathlen_th[N:].max() )
# plt.title('Total error: %3.4f' %error_pathlen, fontsize=12)
# plt.plot((0,maxvalue+1),(0,maxvalue+1), color='gray', lw= 0.5, zorder=0)
# plt.scatter(pathlen_num[N:], pathlen_th[N:], s=10, color='gray', label='Pathlength')
# plt.xlabel('Numerical', fontsize=14)
# plt.ylabel('Analytical', fontsize=14)
# plt.xlim(0.9,maxvalue+0.1)
# plt.ylim(0.9,maxvalue+0.1)
# plt.grid(ls='dotted', zorder=-100)
#
# plt.legend(loc='upper center', fontsize=12, scatterpoints=1, frameon=False)
#
# # 3.4) Validation result for efficiency
# plt.figure()
# plt.title('Total error: %3.4f' %error_effic, fontsize=12)
# plt.plot((0,1),(0,1), color='gray', lw= 0.5, zorder=0)
# plt.scatter(effic_num, effic_th, s=10, color='gray', label='Efficiency')
# plt.xlabel('Numerical', fontsize=14)
# plt.ylabel('Analytical', fontsize=14)
# plt.xlim(0,1)
# plt.ylim(0,1)
# plt.grid(ls='dotted')
#
# plt.legend(loc='upper center', fontsize=12, scatterpoints=1, frameon=False)


plt.show()



#
