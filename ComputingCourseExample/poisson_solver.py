"""
Module to solve the Poisson equation.
"""
import time
import numpy as np
#import matplotlib.pyplot as plt
#is unused
from mpi4py import MPI
import h5py
from cli_helper import read_parameters
from cli_helper import (METHOD_GAUSS, METHOD_JACOBI, PROBLEM_ZERO,
                        PROBLEM_SIN, TERMINATION_ITER, TERMINATION_PREC)
from pde_lib import init_matrices, iterate, iterate_checkerboard


def main(method, problem, matrix_size, termination, termination_value, save_path = None):
    """The main function.
    """
    # TODO: Excercise sheet 9: MPI initialization
    # DONE
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    # Here we initialize the matrices and the disturbance matrix.
    # TODO: Implement a matrix distribution for MPI on exercise sheet 9
    # DONE
    # You can use the init_matrices function to generate submatrices by their
    # corner points.
    # For the single core case the corner points are set to (0,0) and
    # (matrix_size,matrix_size).

    start_column = int((matrix_size - 2) / world_size * rank + 0.5)
    end_column = int((matrix_size - 2) / world_size * (rank + 1) + 2.5)

    matrices, disturbance_matrix = init_matrices(method,
                                                 problem,
                                                 matrix_size,
                                                 (start_column, 0),
                                                 (end_column, matrix_size))

    # Prints the initilized matrices.
    # Do the same with the MPI version and check if the results are the same.

    np.set_printoptions(precision=2, suppress=True)
    #create and collect matrices to print
    print_matrix = get_display_matrix_mpi(matrices[0],
                                          matrix_size,
                                          start_column,
                                          comm)
    print_disturbance_matrix = get_display_matrix_mpi(disturbance_matrix,
                                                      matrix_size,
                                                      start_column,
                                                      comm)

    #print only if rank is 0
    if rank == 0:
        print(print_matrix)
        print(print_disturbance_matrix)

    # The main calculation loop
    start_time = time.time()
    if method == METHOD_GAUSS:
        # TODO: Exercise sheet 10: Implement a MPI version of the gauss solver
        statistics = gauss_solver(matrices,
                                  disturbance_matrix,
                                  termination,
                                  termination_value,
                                  start_column)
    elif method == METHOD_JACOBI:
        # TODO: Exercise sheet 9: Implement a MPI version of the jacobi solver
        statistics = jacobi_solver(matrices,
                                   disturbance_matrix,
                                   termination,
                                   termination_value,
                                   comm)
    end_time = time.time()

    # Print statistics and the result on the cli
    # TODO: This is only for rank 0. Implement on exercise sheet 9
    # DONE
    duration = np.array(end_time-start_time)
    max_duration = np.empty_like(duration)
    comm.Allreduce(duration, max_duration, op = MPI.MAX)

    statistics['duration'] = max_duration
    if rank == 0:
        print('Number of iterations: {}'.format(statistics['iterations']))
        print('Precision: {}'.format(statistics['precision']))
        print('Duration: {}'.format(statistics['duration']))

    #get result matrix and print for rank == 0
    result_display_matrix = get_display_matrix_mpi(matrices[0],
                                                   matrix_size,
                                                   start_column,
                                                   comm)
    if rank == 0:
        print(result_display_matrix)

    # if you wish you can plot the display matrix as well
    # Note: maybe you want to save it as png...
    #plt.matshow(display_matrix)
    #plt.show()
    file = h5py.File(save_path, 'w', driver='mpio', comm=comm)
    file.atomic = True

    dset = file.create_dataset('resultmatrix', (matrix_size, matrix_size), dtype=np.float64)
    left_pad  = 0 if rank == 0              else 1
    right_pad = 0 if rank == world_size - 1 else -1
    dset[start_column + left_pad : end_column + right_pad, :] = matrices[
         0, left_pad : end_column - start_column + right_pad, :]

    dset.attrs.create('iterations', data=statistics['iterations'])
    dset.attrs.create('precision', data=statistics['precision'])
    dset.attrs.create('duration', data=statistics['duration'])
    dset.attrs.create('method',
        data=('gauss' if method == METHOD_GAUSS else 'jacobi'))
    dset.attrs.create('problem',
        data=('zero' if problem == PROBLEM_ZERO else 'sin'))
    dset.attrs.create('term_iterations' if termination == TERMINATION_ITER
        else 'term_precision', data=termination_value)

    file.close()

def get_display_matrix(matrix, matrix_size):
    """Generates a 11x11 matrix for display purposes.

    Args:
        matrix (np.ndarray): The calculation matrix.
        matrix_size (int): The total size of the matrix.

    Returns:
        np.ndarray: The matrix to display
    """
    # we do not want to print the border of the problem
    print_idxs = np.int32(np.round((np.linspace(0,
                                                matrix_size - 1,
                                                11,
                                                endpoint=True))))
    dim1_selection, dim2_selection = np.meshgrid(print_idxs, print_idxs)
    return matrix[dim1_selection, dim2_selection]

def get_display_matrix_mpi(matrix, matrix_size, start_col, comm):
    """TODO: Implement on excercise sheet 9

    Args:
        matrix (np.ndarray): The matrix we want to collect/print
        matrix_size (int): The total matrix size
        start_col (int): Start of current column
        end_col (int): End of current column
        comm (Intracomm): MPI world

    Return:
        recvbuf (np.ndarray): matrix to print if rank==0, else None
    """
    out_size = 11
    world_size = comm.Get_size()
    rank = comm.Get_rank()

    process_scale = (matrix_size-2) / world_size
    output_scale = (matrix_size-1) / (out_size-1)


    if rank == 0:
        start_index = 0

    else:
        start_index = np.ceil(((round(process_scale * rank) + 1) - 0.5)
                              / output_scale)

    if rank == (world_size - 1):
        end_index = np.ceil((matrix_size - 0.5) / output_scale)

    else:
        end_index = np.ceil((round(process_scale * (rank + 1)) + 0.5)
                            / output_scale)

    x_column = np.arange(start_index, end_index, dtype = np.float64)
    dim1_index = np.int32(np.round(x_column * output_scale)) - start_col
    dim2_index = np.int32(np.round((np.linspace(0,
                                                matrix_size - 1,
                                                out_size,
                                                endpoint=True))))
    display_matrix = np.zeros([out_size, out_size])

    display_matrix[np.int32(x_column), :] = matrix[dim1_index, :][:, dim2_index]
    recvbuf = None
    if rank == 0:
        recvbuf = np.empty(display_matrix.shape)
    comm.Reduce(display_matrix, recvbuf, op=MPI.SUM, root=0)
    return recvbuf

def jacobi_solver(matrices, disturbance_matrix, termination, termination_value, comm):
    """TODO: Exercise sheet 9: Implement a MPI version of the jacobi solver
       Done

    Args:
        matrices (np.ndarray): calculation matrices
        disturbance_matrix (np.ndarray): disturbance matrix
        problem (int): problem size
        termination (int): The type of ther termination. Use TERMINATION_ITER or
                           TERMINATION_PREC
        termination_value (int or float): iterations or precision,
                                          see termination
        comm (Intracomm): MPI world

    Return:
        statistics (dict): jacobi algorithm results
    """
    statistics = {'iterations': 0, 'precision': 0}
    matrix_a = 0
    matrix_b = 1
    local_res = np.empty([1], dtype=np.float64)
    global_res = np.empty_like(local_res)

    # set the number of iterations
    if termination == TERMINATION_ITER:
        iterations = termination_value
    else:
        # Note: we set this to 1 for precision.
        # We set this to 0 if precision is reached
        iterations = 1

    while iterations > 0:
        local_res[0] = iterate(matrices,
                               matrix_a,
                               matrix_b,
                               disturbance_matrix,
                               termination,
                               iterations)

        transfer = exchange_halos(comm, matrices[matrix_b])

        matrix_a, matrix_b = matrix_b, matrix_a

        if termination == TERMINATION_PREC:
            comm.Allreduce(local_res, global_res, op=MPI.MAX)
            if global_res[0] < termination_value:
                iterations = 0
        else:
            iterations -= 1
            if iterations == 0:
                comm.Allreduce(local_res, global_res, op=MPI.MAX)
        statistics['iterations'] += 1

        wait_all(transfer)
    statistics['precision'] = local_res[0]
    return statistics

def exchange_halos(comm, matrix):
    """Exchange halo layers between processes

    Args:
        comm (Intracomm): MPI world
        matrix (np.ndarray): calculation matrix

    Return:
        requests (list): send and recieve requests
    """
    if comm.Get_rank() % 2 == 0:
        requests = [send_halo(comm, matrix[-2], +1),
                    recv_halo(comm, matrix[-1], +1),
                    send_halo(comm, matrix[ 1], -1),
                    recv_halo(comm, matrix[ 0], -1)]
    else:
        requests = [recv_halo(comm, matrix[ 0], -1),
                    send_halo(comm, matrix[ 1], -1),
                    recv_halo(comm, matrix[-1], +1),
                    send_halo(comm, matrix[-2], +1)]
    return requests

def send_halo(comm, buffer, direction):
    """send halos between processes if they exist

    Args:
        comm (Intracomm): MPI world
        buffer (np.ndarray): source array
        direction (int): send destination

    Return:
        send (Isend): send request
    """
    dest = comm.Get_rank() + direction
    if dest < 0 or dest >= comm.Get_size():
        return None
    return comm.Isend(buffer, dest=dest, tag=1+direction)

def recv_halo(comm, buffer, direction):
    """recieve halos from processes if they exist

    Args:
        comm (Intracomm): MPI world
        buffer (np.ndarray): target array
        direction (int): sender

    Return:
        recv (Irecv): recieve request
    """
    src = comm.Get_rank() + direction
    if src < 0 or src >= comm.Get_size():
        return None
    return comm.Irecv(buffer, source=src, tag=1-direction)

def wait_all(requests):
    """wait for requests if they exist

    Args:
        requests (list): requests to wait for
    """
    for req in requests:
        if req is not None:
            req.wait()

def wait(request):
    """ Waits for the request if they are not None

    Args:
        request (object): the MPI request to wait for
    """
    if request is not None:
        request.wait()

def gauss_solver(matrices, disturbance_matrix, termination, termination_value,
                 start_column):
    """A MPI version of the gauss seidel solver using a checkerboard template

    Args:
        matrices (np.ndarray): calculation matrices
        disturbance_matrix (np.ndarray): disturbance matrix
        problem (int): problem size
        termination (int): The type of ther termination. Use TERMINATION_ITER or
                           TERMINATION_PREC
        termination_value (int or float): iterations or precision,
                                          see termination
        start_column (int): starting column of the process

    Return:
        statistics (dict): jacobi algorithm results
    """
    # TODO: Exercise sheet 10: Implement a MPI version of the gauss solver
    statistics = {'iterations': 0, 'precision': 0}
    local_res = np.empty([1], dtype=np.float64)
    global_res = np.empty_like(local_res)
    matrix = matrices[0]
    mat_size = matrix.shape[1]
    end_column = start_column + matrix.shape[0]
    #indices{ 0: send-, 1: send+, 2: recv+, 3: recv- }
    combuffers = np.empty([4, (mat_size - 1) // 2], dtype=np.float64)
    requests = [None, None, None, None]

    # set the number of iterations
    if termination == TERMINATION_ITER:
        iterations = termination_value
    else:
        # Note: we set this to 1 for precision.
        # We set this to 0 if precision is reached
        iterations = 1

    while iterations > 0:
        #declare recieves beforehand to increase communication speed
        start_recvs(combuffers, requests)
        #first iterarion
        maxresiduum_1 = iterate_checkerboard(matrix,
                                             disturbance_matrix,
                                             termination,
                                             iterations,
                                             start_column)
        #first communication
        exchange_checkerboard_halos(combuffers,
                                    requests,
                                    matrix,
                                    start_column,
                                    end_column)
        start_recvs(combuffers, requests)
        #second iteration
        maxresiduum_2 = iterate_checkerboard(matrix,
                                             disturbance_matrix,
                                             termination,
                                             iterations,
                                             start_column+1)
        #second communication
        exchange_checkerboard_halos(combuffers,
                                    requests,
                                    matrix,
                                    start_column+1,
                                    end_column+1)
        #result evaluation
        local_res[0] = max(maxresiduum_1, maxresiduum_2)
        if termination == TERMINATION_PREC:
            MPI.COMM_WORLD.Allreduce(local_res, global_res, op=MPI.MAX)
            if global_res[0] < termination_value:
                iterations = 0
        else:
            iterations -= 1
            if iterations == 0:
                MPI.COMM_WORLD.Allreduce(local_res, global_res, op=MPI.MAX)
        statistics['iterations'] += 1
    statistics['precision'] = local_res[0]
    wait_all(requests[:2]) #wait for pending sends
    return statistics

def exchange_checkerboard_halos(buffers, requests, matrix, start_col, end_col):
    """ Exchanges the halos of two intertwined matrices

    Args:
        buffers (np.ndarray): the buffer of the communication
        requests (list): requests for the communication
        matrix (np.ndarray): calculation matrices
        start_col (int): starting column of the process
        end_col (int): end column of the process
    """
    #send to prev column:
    requests[0] = send_copy(buffers[0], matrix[1], start_col+1, -1, requests[0])
    #send to next column:
    requests[1] = send_copy(buffers[1], matrix[-2], end_col-2, +1, requests[1])
    #receive from next column:
    recv_copy(buffers[2], matrix[-1], end_col-1, requests[2])
    #receive from prev column:
    recv_copy(buffers[3], matrix[0], start_col, requests[3])

def start_recvs(buffers, requests):
    """ The first recieves for the communication

    Args:
        buffers (np.ndarray): the buffer of the communication
        requests (list): requests for the communication
    """
    comm = MPI.COMM_WORLD
    requests[2] = recv_halo(comm, buffers[2], +1)
    requests[3] = recv_halo(comm, buffers[3], -1)

def send_copy(buffer, mat_col, offset, direction, old_req):
    """ Sends a part of the column defined by the offset

    Args:
        buffer (np.ndarray): the buffer of the communication
        mat_col (np.ndarray): The matrixcolumn to send
        offset (int): the offset for the checkerboard template
        direction (int): The direction in which the mat_col should be send
        old_req (object): request of the iteration before

    Return:
        send_request (object): send request if sender exists, None else
    """
    comm = MPI.COMM_WORLD
    dest = comm.Get_rank() + direction
    if dest < 0 or dest >= comm.Get_size():
        send_request =  None
    else:
        if old_req is not None:
            old_req.wait()
        column_reference = mat_col[1 + offset%2 : -1 : 2]
        buffer[:len(column_reference)] = column_reference
        send_request = comm.Isend(buffer, dest=dest, tag=1+direction)
    return send_request

def recv_copy(buffer, mat_col, offset, recv_request):
    """ Recieves a part of the column defined by the offset

    Args:
        buffer (np.ndarray): the buffer of the communication
        mat_col (np.ndarray): The matrixcolumn to recieve
        offset (int): the offset for the checkerboard template
        recv_request (object): recieve request
    """
    if recv_request is not None:
        recv_request.wait()
        column_reference = mat_col[1 + offset%2 : -1 : 2]
        column_reference[:] = buffer[:len(column_reference)]

if __name__ == "__main__":
    args = read_parameters()
    main(*args)
