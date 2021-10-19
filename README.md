# Phase field fracture

Python code for phase field fracture problems based on the [muSpectre](https://gitlab.com/muspectre/muspectre) code for FFT-based micromechanics.
This code uses muSpectre (>=0.18.2) installed with MPI, FFTW, and NetCDF.  Numpy and mpi4py are required to run simulations.  Microstructure generation additionally requires scipy, and postprocessing requires parallel-netcdf, matplotlib, and (for one example) pandas.