## TODO list for PathLims


### Priorities...

1. Drop support for Python 2.
2. Clean-up the library files. Remove unnecessary comments, copyright duplicates, etc.
3. f" … " string formatting.
4. Update to the newer packaging and PyPI release standards.
    1. We will use Hatch, at least for now.
    2. Move Matploblib into optional dependencies. Only used to run the examples.
        1. Add `try: import matplotlib` to example files.
    3. Prepare PathLims for conda-sourceforge.
    4. Add instractions for conda users (either `conda install pathlims` or, first install dependencies via conda and then `python -m pip install -e . --no-deps`. Or, release also a *yml* file with preinstallation of the dependencies.
5. What should I do with the Examples/ folder when packaging? Should I integrate it to the wheel, or should I let users to download the examples separately (manually) from the GitHub page? **NO, do not include them into the wheel. Just leave them in the root of the repo, for independent download.**
6. Clean and fix scripts in the Examples/ folder.
7. Integrate PathLims into pyGAlib (?). **I certainly will need to do so, for conda-forge integration. Otherwise I would have to import/install via pip, for a conda package.**
8. Bring weighted network generation and randomization from SiReNetA.
9. Use some *linting* software to double check code.
10. *Suggest your own…*


### This and that...

1. Add the limits for clustering coefficient and the corresponding network generators.*Suggest your own…*


### It would be nice to ...

1. Add functionalities for graph visualization.
2. Finish documentation (use Sphinx for that).
3. *Please, suggest your own…*
