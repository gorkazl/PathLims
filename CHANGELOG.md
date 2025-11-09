### HISTORY OF CHANGES

##### November 9, 2025 (Release of Version 2)

Stable version 2.0 checked, validated and released.

* The library has been reshaped to be compliant with the modern [PyPA specifications](https://packaging.python.org/en/latest/specifications/).
* [Hatch](https://hatch.pypa.io/latest/) was chosen as the tool to build and publish the package. See the *pyproject.toml* file. 
* Bug fixes to adapt to the various changes in NumPy since last release of PathLims.
* Sample and validation scripts in the "*Examples/*" folder revised and adapted to recent changes in Python and NumPy. 


##### November 22, 2019
Stable version 1.0.0 checked, validated and released.

* PathLims has been registered in PyPI ([https://pypi.org/project/pathlims/](https://pypi.org/project/pathlims/)). Direct installation and version management using `pip` is now available.
* Two Jupyter notebooks added for tutorial: *EmpiricalNets_Directed.ipynb* and *EmpiricalNets_Undirected.ipynb.*

##### December 31, 2018
Example scripts have been added and some datasets:

- Two scripts (*EmpiricalNets_Directed.py* and *EmpiricalNets_Undirected.py*) to illustrate how to compare the pathlength and efficiency of real networks to those of ring lattices, random graphs and the true ultra-long and ultra-short limits.
- Four scripts (*UltraLong_Digraphs.py*, *UltraLong_Graphs.py*, *UltraShort_Digraphs.py* and *UltraShort_Graphs.py*) in which ultra-long and ultra-short networks are generated, their pathlength and efficiencies  are numerically calculated, and compared to the theoretical values for corroboration.

##### December 22, 2018
Version 1.0.0b2 (beta) of PathLims has been uploaded.
