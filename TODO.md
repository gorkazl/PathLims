## TODO list for PathLims


### Priorities...

1. ~~Drop support for Python 2.~~
2. ~~Clean-up the library files. Remove unnecessary comments, copyright duplicates, etc.~~
3. ~~f" … " string formatting~~.
4. ~~Replace runtime "prints" by proper warning and error detection.~~
5. Update to the newer packaging and PyPI release standards.
    1. We will use Hatch, at least for now.
    2. Move Matploblib into optional dependencies. Only used to run the examples.
        1. Add `try: import matplotlib` to example files.
    3. Prepare PathLims for conda-sourceforge.
    4. Add instractions for conda users (either `conda install pathlims` or, first install dependencies via conda and then `python -m pip install -e . --no-deps`. Or, release also a *yml* file with preinstallation of the dependencies.
6. Clean, revise and test scripts in the Examples/ folder.
7. Integrate PathLims into pyGAlib (?). **I certainly will need to do so, for conda-forge integration. Otherwise I would have to import/install via pip, for a conda package.**
8. Test, test, test:
    1. Run the example scripts.
    2. Warning and error cases.
9. Update README.md file:
    1. Add version and status indicators at the top. 
    2. Revise installation instructions.


### This and that...

1. Add the limits for clustering coefficient and the corresponding network generators.*Suggest your own…*


### It would be nice to ...

1. Use some *linting* software to double check code.
2. Add functionalities for graph visualization.
3. Finish documentation (use Sphinx for that).
4. *Please, suggest your own…*
