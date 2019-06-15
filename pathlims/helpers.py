# -*- coding: utf-8 -*-
# Copyright (c) 2018 - 2019, Gorka Zamora-López and Romain Brasselet
# <gorka@Zamora-Lopez.xyz>
#
# Released under the Apache License, Version 2.0 (the "License");
# you may not use this software except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
================
HELPER FUNCTIONS
================

Miscelaneous helper functions which are useful to run the examples provided
with the packages. All these functions are taken from the library
'GAlib - Graph Analysis library in Python / NumPy'
(https://github.com/gorkazl/pyGAlib)

LoadFromPajek
    Reads a network from a text file with Pajek format.
Reciprocity
    Computes the fraction of reciprocal links to total number of links.
FloydWarshall
    Computes the pathlength between all pairs of nodes in a network.
Lattice1D_FixLinks
    Generates a 1D ring lattice with given number of links.
RandomGraph
    Generates random graphs with N nodes and L links.

...moduleauthor:: Gorka Zamora-Lopez <galib@zamora-lopez.xyz>

"""

from __future__ import division, print_function# , absolute_import

import numpy as np
import numpy.random
import scipy.special



###############################################################################
def LoadFromPajek(filepath, getlabels=False):
    """Reads a network from a text file with Pajek format.

    Parameters
    ----------
    filepath : string
        The source .net file of the network in Pajek format.
    getlabels : boolean, optional
        If True, the function also reads and returns the labels of the nodes.

    Returns
    -------
    adjmatrix : ndarray of rank-2
        The adjacency matrix of the network.
    labels : list, optional
        If getlabels = True, the function also return a list containing the
        names of the nodes.

    See Also
    --------
    LoadLabels, SaveLabels
    Save2Pajek : Saves a network into a file in Pajek-readable format.

    Notes
    -----
    1. The function automatically determines whether the network is directed,
    undirected and / or weighted.
    2. The returned adjacency matrix is of dtype 'int' or 'float', depending
    on the weights in the file.
    """
    # 0) OPEN THE FILE AND READ THE SIZE OF THE NETWORK
    pajekfile = open(filepath, 'r')
    firstline = pajekfile.readline()
    firstline = firstline.split()
    N = int(firstline[1])

    # 1) READ THE LABELS OF THE NODES IF WANTED
    if getlabels:
        labels = []

        # Security check, make sure that labels of nodes are listed in file
        line = pajekfile.readline()
        if line.split()[0] != '1':
            pajekfile.seek(1)
            print('LoadFromPajek() warning: No labels found to read.')

        # If labels are in file continue reading the labels.
        else:
            # If labels are wrapped in between quotes
            try:
                idx1 = line.index('"') + 1
                # Add the first label
                idx2 = line[idx1:].index('"')
                label = line[idx1:idx1+idx2]
                labels.append(label)

                # And now read the labels for the rest of the nodes
                for i in range(1,N):
                    line = pajekfile.readline()
                    idx1 = line.index('"') + 1
                    idx2 = line[idx1:].index('"')
                    label = line[idx1:idx1+idx2]
                    labels.append(label)

            # Otherwise, make a wild guess of what the label is
            except ValueError:
                # Add the first label
                label = line.split()[1]
                labels.append(label)

                # And now read the labels of the rest of the nodes
                for i in range(1,N):
                    line = pajekfile.readline()
                    label = line.split()[1]
                    labels.append(label)

    # 2) READ THE LINKS AND CREATE THE ADJACENCY MATRIX
    # 2.1) Find out whether the network is directed or undirected
    # while loop to skip empty lines if needed or the lines of the labels
    done = False
    while not done:
        line = pajekfile.readline()
        if line[0] == '*':
            if 'Edges' in line:
                directed = False
            elif 'Arcs' in line:
                directed = True
            else:
                print('Could not find whether network is directed or undirected')
                break
            done = True

    # 2.2) Read the first line contining a link
    line = pajekfile.readline()
    line = line.split()

    # If link information is BINARY, just read the adjacency list links
    if len(line) == 2:
        # 2.3) Declare the adjacency matrix and include the first link
        adjmatrix = np.zeros((N,N), np.uint8)
        i = int(line[0]) - 1
        j = int(line[1]) - 1
        adjmatrix[i,j] = 1
        if not directed:
            adjmatrix[j,i] = 1

        # 2.4) Include the rest of the links
        for line in pajekfile:
            i, j = line.split()
            i = int(i) - 1
            j = int(j) - 1
            adjmatrix[i, j] = 1
            if not directed:
                adjmatrix[j, i] = 1

    # If the link information is WEIGHTED, read the weighted links
    elif len(line) == 3:
        # 2.3) Find whether link weights are integer or floating poing
        i, j, aij = line
        outdtype = np.int
        try:
            outdtype(aij)
        except ValueError:
            outdtype = np.float

        # 2.4) Declare the adjacency matrix and include the first link
        adjmatrix = np.zeros((N, N), outdtype)
        i = int(i) - 1
        j = int(j) - 1
        adjmatrix[i, j] = outdtype(aij)
        if not directed:
            adjmatrix[j, i] = outdtype(aij)

        # 2.5) Read the rest of the file and fill-in the adjacency matrix
        for line in pajekfile:
            i, j, aij = line.split()
            i = int(i) - 1
            j = int(j) - 1
            adjmatrix[i, j] = outdtype(aij)
            if not directed:
                adjmatrix[j, i] = adjmatrix[i, j]

    # 3) CLOSE FILE AND RETURN RESULTS
    pajekfile.close()

    if getlabels:
        return adjmatrix, labels
    else:
        return adjmatrix

def Reciprocity(adjmatrix):
    """Computes the fraction of reciprocal links to total number of links.

    Both weighted and unweighted input matrices are permitted. Weights
    are ignored for the calculation.

    Parameters
    ----------
    adjmatrix : ndarray of rank-2
        The adjacency matrix of the network.

    Returns
    -------
    reciprocity : float
        A scalar value between 0 (for acyclic directed networks) and 1 (for
        fully reciprocal).
    """
    # 0) PREPARE FOR COMPUTATIONS
    adjmatrix = adjmatrix.astype('bool')

    # 1) COMPUTE THE RECIPROCITY
    L = adjmatrix.sum()
    if L == 0:
        reciprocity = 0
    else:
        # Find the assymmetric links
        # Rest = np.abs(adjmatrix - adjmatrix.T)
        Rest = np.abs(adjmatrix ^ adjmatrix.T)
        Lsingle = 0.5*Rest.sum()
        reciprocity = np.float(L-Lsingle) / L

    return reciprocity

def FloydWarshall(adjmatrix, weighted_dist = False):
    """Computes the pathlength between all pairs of nodes in a network..

    Parameters
    ----------
    adjmatrix : ndarray of rank-2
        The adjacency matrix of the network.
    weighted_dist : boolean, optional
        True if the path distance shall be computed considering the weights
        of the links, False, otherwise. If the adjmatrix is a weighted
        network but'weighted = False', the unweighted graph distance is
        computed.

    Returns
    -------
    distmatrix : ndarray of rank-2
        The pairwise distance matrix dij of the shortest path between
        nodes i and j.
    """
    # Prepare for computations
    if weighted_dist:
        distmatrix = np.where(adjmatrix == 0, np.inf, adjmatrix)
    else:
        distmatrix = np.where(adjmatrix == 0, np.inf, 1)

    # Check whether the network is directed or undirected
    recip = Reciprocity(adjmatrix)

    N = len(adjmatrix)
    # Run the Floyd-Warshall algorithm - Undirected networks
    if recip == 1.0:
        for k in range(N):
            for i in range(N):
                for j in range(i,N):
                    d = distmatrix[i,k] + distmatrix[k,j]
                    if distmatrix[i,j] > d:
                        distmatrix[i,j] = d
                        distmatrix[j,i] = d

    # Run the Floyd-Warshall algorithm - directed networks
    else:
        for k in range(N):
            for i in range(N):
                for j in range(N):
                    d = distmatrix[i,k] + distmatrix[k,j]
                    if distmatrix[i,j] > d:
                        distmatrix[i,j] = d

    return distmatrix

def Lattice1D_FixLinks(N,L):
    """Generates a 1D ring lattice with given number of links.

    Because the total number of links L is fixed, the resulting ring is not a
    perfect regular lattice in which all nodes have exactly the same degree.
    The result is quasi-regular. The largest degree found will be kmax = <k> + 2
    and the smallest degree kmin = <k> - 2.

    Parameters
    ----------
    N : integer
        Size of the network (number of nodes).
    L : integer
        Number of links the resulting network must have.

    Returns
    -------
    adjmatrix : ndarray of rank-2 and integer type.
        The adjacency matrix of the 1-dimensional lattice.

    See Also
    --------
    Lattice1D : Generates regular ring lattices (all nodes have same degree)
    """
    # 0) SECURITY CHECK
    if L > int(0.5*N*(N-1)):
        raise ValueError( "Largest number of links 1/2 * N*(N-1) =", int(0.5*N*(N-1)) )

    if L == 0:
        return np.zeros((N,N), np.uint8)

    adjmatrix = np.zeros((N,N), np.uint8)

    # 1.2) Use numpy.roll() to copy rotated version of a pattern row
    counter = 0
    finished = False
    for k in range(1,N):
        if finished: break
        row = np.zeros(N,np.uint8)
        row[k] = 1

        for i in range(N):
            adjmatrix[i] += np.roll(row,i)
            counter += 1
            if counter == L:
                finished = True
                break

    return adjmatrix + adjmatrix.T

def RandomGraph(N, L, directed=False, selfloops=False):
    """Generates random graphs with N nodes and L links.

    Similar to an Erdos-Renyi (ER) graph with probability p = rho, where
    rho is the density of links. In ER graphs the total number of links
    varies in different realizations what is unsuitable to compare empirical
    networks with their random counterparts. RandomGraph() allows to create
    random graphs that always have the same number of links. The precise
    formula for rho depends on the options given. For an undirected graph
    with no self-loops allowed, rho = 1/2 * L / (N*(N-1)).

    Parameters
    ----------
    N : integer
        The size of the network (number of nodes).
    L : integer
        Number of links of the resulting random network.
    directed : Boolean
        True if a directed graph is desired. False, for an undirected graph.
    selfloops: Boolean
        True if self-loops are allowed, False otherwise.

    Returns
    -------
    adjmatrix : ndarray of rank-2, size NxN and dtype = int.
        The adjacency matrix of the generated random graph.

    Notes
    -----
    Make sure to specify the right number of links for the type of graph
    desired. Keep in mind that directed graphs have twice as many links
    (arcs) as graphs (edges).

    See Also
    --------
    ErdosRenyiGraph : Random graphs with given link probability.
    """
    # 0) SECURITY CHECKS. Make sure L is not too large.
    if directed:
        if selfloops:
            maxL = N**2
            if L > maxL:
                raise ValueError( "L out of bounds, max(L) = N**2 =", maxL )
        else:
            maxL = N*(N-1)
            if L > maxL:
                raise ValueError( "L out of bounds, max(L) = N*(N-1) =", maxL )
    else:
        if selfloops:
            maxL = 0.5*N*(N+1)
            if L > maxL:
                raise ValueError( "L out of bounds, max(L) = 1/2*N*(N+1) =", maxL )
        else:
            maxL = 0.5*N*(N-1)
            if L > maxL:
                raise ValueError( "L out of bounds, max(L) = 1/2*N*(N-1) =", maxL )

    # 1) INITIATE THE MATRIX AND HELPERS
    adjmatrix = np.zeros((N,N), np.uint8)
    counter = 0

    # 2) GENERATE THE MATRIX
    while counter < L:
        # 2.1) Pick up two nodes at random
        source = int(N * numpy.random.rand())
        target = int(N * numpy.random.rand())

        # 2.2) Check if they can be linked, otherwise look for another pair
        if adjmatrix[source,target] == 1: continue
        if not selfloops and source == target: continue

        # 2.3) If the nodes are linkable, place the link
        adjmatrix[source,target] = 1
        if not directed:
            adjmatrix[target,source] = 1

        counter += 1

    return adjmatrix


##
