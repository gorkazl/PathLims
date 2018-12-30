# -*- coding: utf-8 -*-
#
# Copyright (c) 2018 - 2019, Gorka Zamora-López <gorka@Zamora-Lopez.xyz>
#
# Released under the Apache License, Version 2.0 (the "License");
# you may not use this software except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# REFERENCE AND CITATION
# When using PathLims please cite:
#
# G. Zamora-Lopez & R. Brasselet *Sizing the length of complex networks*
# arXiv:1810.12825 (2018).
#
#

"""
This script is an example to use and validate the generation and analytic
estimation of the ultra-short boundaries for directed graphs. We will:

1. Generate ultra-short directed graphs,
2. Numerically compute their pathlength and efficiency,
3. Analytically estimate their pathlength and efficiency, and
4. Compare the results.
"""
from __future__ import division, print_function, absolute_import

# Standard library imports
from timeit import default_timer as timer
# Third party imports
import matplotlib.pyplot as plt
import numpy as np
# Local imports
import pathlims
from pathlims.limits import Pathlen_USdigraph, Effic_USdigraph
from pathlims.generators import USdigraph
from pathlims.helpers import FloydWarshall


################################################################################
# 0) PREPARE FOR THE CALCULATIONS
# Type of ultra-short graph, 'Random' or 'RichClub'
ustype = 'Random'

# Basic properties of the digraphs
N = 8
Lmax = int( N*(N-1) )
Llist = np.arange(Lmax+1)
nL = len(Llist)


# 1) DO THE CALCULATIONS
pathlen_th = np.zeros(nL, np.float)
pathlen_num = np.zeros(nL, np.float)
effic_th = np.zeros(nL, np.float)
effic_num = np.zeros(nL, np.float)
time1 = timer()
for L in range(nL):

    # 1.1) Generate the ultra-short graph and compute distance / efficiency
    ## Note: USdigraph() function calls different models according to L and
    ## parameters 'uscase' or 'onlyconnected'. Check function's help for further
    ## information.
    ## If L < N generates a disconnected graph, and a connected one otherwise.
    net = USdigraph(N,L, uscase=ustype, onlyconnected=True)
    dij = FloydWarshall(net)
    pathlen_num[L] = dij.sum() - dij.trace()
    eij = 1./dij
    effic_num[L] = eij.sum() - eij.trace()

    # 1.2) Calculate the results analitically
    ## Note: Pathlen_USdigraph() returns 'inf', for L < N-1.
    pathlen_th[L] = Pathlen_USdigraph(N,L)
    effic_th[L]   = Effic_USdigraph(N,L)

# Normalise the numerical results
pathlen_num /= Lmax
effic_num /= Lmax

time2 = timer()
print(time2 - time1, 'seconds')


# 2) SUMMARISE THE RESULTS
diff_pathlen = abs(pathlen_num[N:] - pathlen_th[N:])
error_pathlen = diff_pathlen.sum()
print('Pathlength. Total error:', error_pathlen )

diff_effic = abs(effic_num - effic_th)
error_effic = diff_effic.sum()
print('Efficiency. Total error:', error_effic )


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

plt.legend(loc='upper center', fontsize=12, scatterpoints=1, frameon=False)

# 3.2) Plot comparison for efficiency
plt.figure()
plt.title('Total error: %3.4f' %error_effic, fontsize=12)
plt.scatter(Llist, effic_num, s=10, color='#d62728', label='Numerical')
plt.plot(Llist, effic_th, color='gray', zorder=0, label='Theoretical')
plt.xlabel('Number of edges', fontsize=14)
plt.ylabel('Efficiency', fontsize=14)
plt.xlim(-1,nL+1)
plt.ylim(-0.1,1.1)
plt.grid(ls='dotted')

plt.legend(loc='upper center', fontsize=12, scatterpoints=1, frameon=False)


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
