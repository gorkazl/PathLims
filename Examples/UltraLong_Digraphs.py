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
estimation of the ultra-long boundaries for directed graphs. We will:

1. Generate ultra-long directed graphs,
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
from pathlims.limits import ( Pathlen_ULdigraph_Range1_MBS,
                            Pathlen_ULdigraph_Range1_Approx,
                            Pathlen_ULdigraph_Intermediate,
                            Pathlen_ULdigraph_Range2,
                            Effic_ULdigraph_Range1_MBS,
                            Effic_ULdigraph_Intermediate,
                            Effic_ULdigraph_Disconnected )
from pathlims.generators import (ULdigraph_Connected_Range1_MBS,
                            ULdigraph_Connected_Intermediate,
                            ULdigraph_Connected_Range2,
                            ULdigraph_Disconnected_Range1,
                            ULdigraph_Disconnected_Range2 )
from pathlims.helpers import FloydWarshall


################################################################################
# 0) PREPARE FOR THE CALCULATIONS
# Basic properties of the graphs.
N = 10
Lmax = int( N*(N-1) )
Llist = np.arange(Lmax+1)
nL = len(Llist)

# 1) DO THE CALCULATIONS -- CONNECTED GRAPHS
time1 = timer()

# 1.1) Calculate results for the range-1, when N <= L < (N-1) + 1/2 N(N-1)
# In this case, we provide the results only for given values of L such that
# L = N + 1/2 M(M-1) with M = 1, 2, ..., N-1. Results for remaining values
# of L will be treated in the future.
# For analytical estimation of the UL limit in digraphs for any value of
# L < (N-1) + 1/2 N(N-1), see also the approximated result given by function
# Pathlen_ULdigraph_Range1_Approx()
Mlist1 = np.arange(1,N)
nM = len(Mlist1)
Llist1 = np.zeros(nM, np.uint)
pathlen_num1 = np.zeros(nM, np.float)
pathlen_th1 = np.zeros(nM, np.float)
effic_num_con1 = np.zeros(nM, np.float)
effic_th_con1 = np.zeros(nM, np.float)

for counter,M in enumerate(Mlist1):
    LM = int( N + 0.5*M*(M-1) )
    Llist1[counter] = LM

    # Generate the connected ultra-long graph and compute distance / efficiency
    net = ULdigraph_Connected_Range1_MBS(N,M)
    dij = FloydWarshall(net)
    pathlen_num1[counter] = ( dij.sum() - dij.trace() ) / Lmax
    eij = 1./dij
    effic_num_con1[counter] = ( eij.sum() - eij.trace() ) / Lmax

    # 1.2) Calculate the results analitically
    pathlen_th1[counter] = Pathlen_ULdigraph_Range1_MBS(N,M)
    effic_th_con1[counter] = Effic_ULdigraph_Range1_MBS(N,M)

# 1.2) Calculate the results for the intermediate configuration, when
# L = (N-1) + 1/2 N(N-1)
Linter = int( N-1 + 0.5*N*(N-1) )
net = ULdigraph_Connected_Intermediate(N)
dij = FloydWarshall(net)
eij = 1./dij
pathlen_num_inter = ( dij.sum() - dij.trace() ) / Lmax
effic_num_inter = ( eij.sum() - eij.trace() ) / Lmax

pathlen_th_inter = Pathlen_ULdigraph_Intermediate(N)
effic_th_inter = Effic_ULdigraph_Intermediate(N)

# 1.3) Calculate results for the range-2, when L > (N-1) + 1/2 N(N-1)
L2min = int( (N+1) + 0.5*N*(N-1) )
Llist2 = np.arange(L2min, Lmax+1)
nL2 = len(Llist2)
pathlen_num2 = np.zeros(nL2, np.float)
pathlen_th2 = np.zeros(nL2, np.float)
effic_num_con2 = np.zeros(nL2, np.float)
effic_th_con2 = np.zeros(nL2, np.float)

for counter,L in enumerate(Llist2):
    # Generate the connected ultra-long graph and compute distance / efficiency
    net = ULdigraph_Connected_Range2(N,L)
    dij = FloydWarshall(net)
    pathlen_num2[counter] = ( dij.sum() - dij.trace() ) / Lmax
    eij = 1./dij
    effic_num_con2[counter] = ( eij.sum() - eij.trace() ) / Lmax

    # Calculate the results analitically
    pathlen_th2[counter] = Pathlen_ULdigraph_Range2(N,L)
    #effic_th_con2[L] = Effic_ULgraph(N,L, connected=True)


# 2) DO THE CALCULATIONS -- DISCONNECTED GRAPHS
# 2.1) Calculate results for the range-1, when L <= 1/2 N(N-1)
Lhalf = int( 0.5*N*(N-1) )
Llist_disco1 = np.arange(Lhalf+1)
nL1 = len(Llist_disco1)
effic_num_disco1 = np.zeros(nL1, np.float)
effic_th_disco1 = np.zeros(nL1, np.float)

for L in range(nL1):
    # Generate the connected ultra-long graph and compute distance / efficiency
    net = ULdigraph_Disconnected_Range1(N,L)
    dij = FloydWarshall(net)
    eij = 1./dij
    effic_num_disco1[L] = ( eij.sum() - eij.trace() ) / Lmax

    # Calculate the results analitically
    effic_th_disco1[L] = Effic_ULdigraph_Disconnected(N,L)

# 2.2) Calculate results for the range-1, when L >= 1/2 N(N-1)
# In this case, we provide the results only for given values of L such that
# L = Lo/2 + 1/2 M(M-1) with M = 1, 2, ..., N. Results for remaining values
# of L will be treated in the future.

Mlist = np.arange(2,N+1)
nM = len(Mlist)
Llist_disco2 = np.zeros(nM, np.uint)
effic_num_disco2 = np.zeros(nM, np.float)
effic_th_disco2 = np.zeros(nM, np.float)

for counter,M in enumerate(Mlist):
    # Get number of edges in an M-complete graph of order M
    LM = Lhalf + int( 0.5*M*(M-1) )
    Llist_disco2[counter] = LM

    # Generate the M-complete graph and compute its efficiency
    net = ULdigraph_Disconnected_Range2(N,M)
    dij = FloydWarshall(net)
    eij = 1./dij
    effic_num_disco2[counter] = ( eij.sum() - eij.trace() ) / Lmax

    # Calculate the results analitically
    effic_th_disco2[counter] = Effic_ULdigraph_Disconnected(N,LM)

time2 = timer()
print(time2 - time1, 'seconds')


# 3) SUMMARISE THE RESULTS
diff_pathlen1 = abs(pathlen_num1 - pathlen_th1)
error_pathlen1 = diff_pathlen1.sum()
print('Pathlength (Range-1). Total error:', error_pathlen1 )

diff_pathlen2 = abs(pathlen_num2 - pathlen_th2)
error_pathlen2 = diff_pathlen2.sum()
print('Pathlength (Range-2). Total error:', error_pathlen2 )

diff_effic_con1 = abs(effic_num_con1 - effic_th_con1)
error_effic_con1 = diff_effic_con1.sum()
print('Efficiency (Connected, Range-1). Total error:', error_effic_con1 )

# diff_effic_con2 = abs(effic_num_con2 - effic_th_con2)
# error_effic_con2 = diff_effic_con2.sum()
# print('Efficiency (Connected, Range-2). Total error:', error_effic_con2 )

diff_effic_disco1 = abs(effic_num_disco1 - effic_th_disco1)
error_effic_disco1 = diff_effic_disco1.sum()
print('Efficiency (Disconnected, Range-1). Total error:', error_effic_disco1 )

diff_effic_disco2 = abs(effic_num_disco2 - effic_th_disco2)
error_effic_disco2 = diff_effic_disco2.sum()
print('Efficiency (Disconnected, Range-2). Total error:', error_effic_disco2 )



# 4) PLOT THE RESULTS
# 4.1) Plot comparison for pathlength
plt.figure()
# Results for the connected range-1
plt.scatter(Llist1, pathlen_num1, s=10, color='#1f77b4', label='Numerical')
plt.plot(Llist1, pathlen_th1, color='gray', zorder=0, label='Theoretical')
# Results for the connected intermediate case, between range-1 and range-2
plt.scatter(Linter, pathlen_num_inter, s=10, color='#1f77b4')
plt.scatter(Linter, pathlen_th_inter, marker= 'x', color='gray', zorder=0)
# Results for the connected range-2
plt.scatter(Llist2, pathlen_num2, s=10, color='#1f77b4')
plt.plot(Llist2, pathlen_th2, color='gray', zorder=0)

plt.xlabel('Number of edges', fontsize=14)
plt.ylabel('Average pathlength', fontsize=14)
plt.xlim(-1,nL+1)
# plt.ylim(1,maxvalue)
plt.grid(ls='dotted')

plt.legend(loc='upper center', fontsize=10, scatterpoints=1, frameon=False)

# 4.2) Plot comparison for efficiency
plt.figure()
# Results for the connected range-1
plt.scatter(Llist1, effic_num_con1, s=10, color='#1f77b4', label='Numerical (connected)')
plt.plot(Llist1, effic_th_con1, color='gray', zorder=0, label='Theoretical (connected)')
# Results for the connected intermediate case, between range-1 and range-2
plt.scatter(Linter, effic_num_inter, s=10, color='#1f77b4')
plt.scatter(Linter, effic_th_inter, marker='x', color='gray', zorder=0)
# Results for the connected range-2

# Results for the disconnected digraphs
plt.scatter(Llist_disco1, effic_num_disco1, s=10, color='#d62728', label='Numerical (disconnected)', zorder=20)
plt.plot(Llist_disco1, effic_th_disco1, '--', color='gray', label='Theoretical (disconnected)', zorder=10)
plt.scatter(Llist_disco2, effic_num_disco2, s=10, color='#d62728', zorder=20)
plt.plot(Llist_disco2, effic_th_disco2, '--', color='gray', zorder=10)

plt.xlabel('Number of edges', fontsize=14)
plt.ylabel('Efficiency', fontsize=14)
plt.xlim(-1,nL+1)
plt.ylim(-0.1,1.1)
plt.grid(ls='dotted')

plt.legend(loc='upper left', fontsize=10, scatterpoints=1, frameon=False)


plt.show()



#
