# Linear System Library
> Methods of Scientific Computing Project for University Milano Bicocca. 2023-2024. Grade: 30

[![Download Relazione PDF](https://img.shields.io/badge/Download%20Relazione-PDF-lime.svg?style=for-the-badge)](https://github.com/Zeptogram/LinearSystemLibrary/releases/download/final/MCS_Relazione_Progetto_1_Biancini_Mattia_865966_Gargiulo_Elio_869184.pdf)
[![Download Presentazione PDF](https://img.shields.io/badge/Download%20Presentazione-PDF-orange.svg?style=for-the-badge)](https://github.com/Zeptogram/LinearSystemLibrary/releases/download/final/Progetto.1.Bis.MCS.Final.pdf)
[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)

Mini Library for the implementation of algorithms made for solving linear systems, using iterative methods for sparse matrices:
- [Jacobi](https://en.wikipedia.org/wiki/Jacobi_method)
- [Gauss Seidel](https://en.wikipedia.org/wiki/Gauss%E2%80%93Seidel_method)
- [Gradient](https://en.wikipedia.org/wiki/Gradient_descent#Solution_of_a_linear_system)
- [Conjugate Gradient](https://en.wikipedia.org/wiki/Conjugate_gradient_method)

## Project Overview

The general structure of the project consists of two main Python files:
- **main.py**: This file contains functions for importing and loading matrices via SciPy, functions for writing the results obtained from executing the solution methods to a file, and the actual use of the library implemented in `methods.py`.
- **methods.py**: This file implements the core functions of the library. Specifically, it includes the implementation of the four solution methods, supplementary functions that help verify/simplify method operations (e.g., checking for row diagonal dominance), and debugging functions.

The project is divided into two main classes: `main.py`, the entry point of the program where sparse matrices in .mtx format can be imported via SciPy, and `methods.py`, which is the actual core of the library.

For libraries handling the data structures of Matrices and Vectors, we have adopted:
- **NumPy**: Provides very efficient tools for matrix operations and the management of dense matrix and vector data structures.
- **SciPy**: Provides a data structure to manage sparse matrices in memory in a manner similar to MatLab.

Download the documentation for detailed info about the whole project (only in Italian). 

## Authors

- Mattia Biancini
- Elio Gargiulo
