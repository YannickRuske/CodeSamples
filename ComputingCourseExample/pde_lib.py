"""
This modules contains functions for Poisson PDE solvers.
You do not need to change any of the functions here to solve the exercises.
"""
from typing import Tuple
import numpy as np
from numba import njit
from cli_helper import METHOD_JACOBI, PROBLEM_ZERO, PROBLEM_SIN, TERMINATION_PREC

def init_calculation_matrices(method: int,
                              problem: int,
                              matrix_size: int,
                              start_corner: Tuple[int],
                              end_corner: Tuple[int]):
    """Initiaizes the calculation matrices according to the specific problem.

    Args:
        method (int): The method to solve since jacobi needs two matrices.
        problem (int): The problem to solve to initialize the matrices.
        matrix_size (int): The total size of the matrix.
        start_corner (Tuple[int]): The start corner of the matrix for the rank.
        end_corner (Tuple[int]): The end corner of the matrix for the rank.

    Returns:
        np.ndarray: The allocated and initialized matrices.
    """
    sc_dim1, sc_dim2 = start_corner
    ec_dim1, ec_dim2 = end_corner
    dim1 = ec_dim1 - sc_dim1
    dim2 = ec_dim2 - sc_dim2

    number_of_matrices = 2 if method == METHOD_JACOBI else 1
    matrices = np.zeros((number_of_matrices, dim1, dim2))

    if problem == PROBLEM_ZERO:
        slope = 1.0 / (matrix_size - 1)
        for m_idx in range(number_of_matrices):
            if sc_dim1 == 0:
                matrices[m_idx, 0, :] = np.linspace(-sc_dim2 * slope + 1.0,
                                                    -ec_dim2 * slope + 1.0,
                                                    dim2, endpoint=False)
            if ec_dim1 == matrix_size:
                matrices[m_idx, -1, :] = np.linspace(sc_dim2 * slope,
                                                     ec_dim2 * slope,
                                                     dim2, endpoint=False)
            if sc_dim2 == 0:
                matrices[m_idx, :, 0] = np.linspace(-sc_dim1 * slope + 1.0,
                                                    -ec_dim1 * slope + 1.0,
                                                    dim1, endpoint=False)
            if ec_dim2 == matrix_size:
                matrices[m_idx, :, -1] = np.linspace(sc_dim1 * slope,
                                                     ec_dim1 * slope,
                                                    dim1, endpoint=False)
    return matrices

def init_disturbance(problem: int,
                     matrix_size: int,
                     start_corner: Tuple[int],
                     end_corner: Tuple[int]):
    """Initializes the disturbance (f(x,y)) as a matrix.
    For memory optimization call this function before init_matrices
    Args:
        problem (int): The problem to solve. Use PROBLEM_SIN or PROBLEM_ZERO
        matrix_size (int): The size of the complete matrix.
        start_corner (Tuple[int]): The start corner of the sub matrix for the rank.
        end_corner (Tuple[int]): The end corner of the sub matrix of the rank.

    Returns:
        np.ndarray: A matrix with all values filled according to the problem
                    specific disturbance function.
    """
    sc_dim1, sc_dim2 = start_corner
    ec_dim1, ec_dim2 = end_corner
    dim1 = ec_dim1 - sc_dim1
    dim2 = ec_dim2 - sc_dim2

    if problem == PROBLEM_SIN:
        slope = 1.0 / (matrix_size - 1)
        # assuming that we call init_matrices after this, the memory
        # consumption should be fine with this method.
        # If memory is a limitation one should consider to change this
        # to for loops which have a poor performance in python.
        # Since this function is called once, it should not make
        # a big difference anyway.
        x = np.pi * np.linspace(sc_dim1 * slope, ec_dim1 * slope, dim1, endpoint=False)
        y = np.pi * np.linspace(sc_dim2 * slope, ec_dim2 * slope, dim2, endpoint=False)
        X, Y = np.meshgrid(y, x) # note that numpy follows a[y, x]
        disturbance_matrix = 2*np.pi**2*np.sin(X)*np.sin(Y)*slope*slope
    else:
        disturbance_matrix = np.zeros((dim1, dim2))
    return disturbance_matrix

def init_matrices(method: int,
                  problem: int,
                  matrix_size: int,
                  start_corner: Tuple[int],
                  end_corner: Tuple[int]):
    """Initializes the matrices for calculation and disturbance.
    You can choose a start and end corner to generate sub matrices for the different
    ranks in MPI Applications.

    The calculation matrix is of the dimension 1xn1xn2 for the Gauss-Seidel method and
    2xn1xn2 for the Jacobi method. n1 and n2 are calculatet from the two corners of
    the matrix.

    Args:
        method (int): The solver method. Use either METHOD_GAUSS or METHOD_JACOBI
        problem (int): The problem to solve. Use either PROBLEM_ZERO or PROBLEM_SIN
        matrix_size (int): The complete size of the matrix.
        start_corner (Tuple[int]): The start corner of the matrix.
        end_corner (Tuple[int]): The end corner of the matrix.

    Returns:
        (np.ndarray, np.ndarray): The calculation matrix and the matrix of the disturbance.
    """
    disturbance_matrix = init_disturbance(problem,
                                          matrix_size,
                                          start_corner,
                                          end_corner)
    calculation_matrix = init_calculation_matrices(method,
                                                   problem,
                                                   matrix_size,
                                                   start_corner,
                                                   end_corner)
    return calculation_matrix, disturbance_matrix

@njit
def iterate(matrices, matrix_in, matrix_out, disturbance_matrix, termination, iteration):
    """Runs one iteration of the moving differentiation star.

    Args:
        matrices (np.ndarray): The calculation matrices
        matrix_in (int): The index of the input matrix
        matrix_out (int): The index of the output matrix
        disturbance_matrix (np.ndarray): The disturbance matrix.
        termination (int): The type of ther termination.
                           Use TERMINATION_ITER or TERMINATION_PREC
        iteration (int): The current iteration.

    Returns:
        float: The maximum of the residuum. (A value for the error.)
               Is 0 for all iterations beside the last one for TERMINATION_ITER
    """
    maxresiduum = 0
    dim_x, dim_y = matrices[matrix_in].shape
    for idx in range(1, dim_x - 1):
        for idy in range(1, dim_y - 1):
            star = 0.25 * (matrices[matrix_in, idx - 1, idy] +
                           matrices[matrix_in, idx + 1, idy] +
                           matrices[matrix_in, idx, idy - 1] +
                           matrices[matrix_in, idx, idy + 1])

            star = star + disturbance_matrix[idx, idy]

            if termination == TERMINATION_PREC or iteration == 1:
                residuum = np.abs(matrices[matrix_in, idx, idy] - star)
                maxresiduum = residuum if residuum > maxresiduum else maxresiduum

            matrices[matrix_out, idx, idy] = star
    return maxresiduum

@njit
def iterate_checkerboard(matrix, disturbance_matrix, termination, iteration, offset):
    """Runs one iteration of the moving differentiation star.

    Args:
        matrix (np.ndarray): The calculation matrix
        disturbance_matrix (np.ndarray): The disturbance matrix.
        termination (int): The type of ther termination.
                           Use TERMINATION_ITER or TERMINATION_PREC
        iteration (int): The current iteration.
        offset (int): offset of first column

    Returns:
        float: The maximum of the residuum. (A value for the error.)
               Is 0 for all iterations beside the last one for TERMINATION_ITER
    """
    maxresiduum = 0
    dim_x, dim_y = matrix.shape
    for idx in range(1, dim_x - 1):
        for idy in range(1 + (offset + idx)%2, dim_y - 1,2):
            star = 0.25 * (matrix[idx - 1, idy] +
                           matrix[idx + 1, idy] +
                           matrix[idx, idy - 1] +
                           matrix[idx, idy + 1])

            star = star + disturbance_matrix[idx, idy]

            if termination == TERMINATION_PREC or iteration == 1:
                residuum = np.abs(matrix[idx, idy] - star)
                maxresiduum = residuum if residuum > maxresiduum else maxresiduum

            matrix[idx, idy] = star
    return maxresiduum