"""
Module to load parameters from CLI. Furthermore global constants are set for more
code readability
"""
import argparse

METHOD_JACOBI = 0
METHOD_GAUSS = 1
PROBLEM_ZERO = 0
PROBLEM_SIN = 1
TERMINATION_ITER = 0
TERMINATION_PREC = 1

def read_parameters():
    """Reads the parameters from the CLI.

    Returns:
        [type]: The argparse object.
    """
    parser = create_parser()
    args = parser.parse_args()

    matrix_size = args.system_size

    method = METHOD_JACOBI if args.method == 'jacobi' else METHOD_GAUSS
    problem = PROBLEM_SIN if args.problem == 'sin' else PROBLEM_ZERO
    termination = TERMINATION_ITER if args.precision is None else TERMINATION_PREC
    termination_value = args.iterations if termination == TERMINATION_ITER else args.precision
    savepath = args.savepath if args.savepath != '' else None

    return method, problem, matrix_size, termination, termination_value, savepath

def create_parser():
    """Create the parser.

    Returns:
        [type]: The parser
    """
    parser = argparse.ArgumentParser(description='Programm to solve partial\
                                     differential equations.')
    parser.add_argument('system_size', type=check_system_size,
                        help='The system size. Has to be 10 or larger.')
    parser.add_argument('method', type=str, choices=['jacobi', 'gauss'],
                        help='The method to solve the system.')
    parser.add_argument('problem', type=str, choices=['zero', 'sin'],
                        help='The problem to solve.')

    parser.add_argument('-s', '--savepath', type=str, required=False, default='',
                        help='The path to a hdf5 file to save the simulation data')
    
    stop_group = parser.add_mutually_exclusive_group(required=True)
    stop_group.add_argument('-p', '--precision', type=check_positive, required=False,
                        help='The precision to reach.')
    stop_group.add_argument('-i', '--iterations', type=check_positive_int, required=False,
                        help='The maximum number of iterations.')
    return parser

def check_system_size(value):
    """Checks if number is a positive int and larger 10.

    Args:
        value (any): The value to check

    Raises:
        argparse.ArgumentTypeError: If the number is not larger 10.

    Returns:
        int: The value.
    """
    value = check_positive_int(value)
    if value < 11:
        raise argparse.ArgumentTypeError("{} is to small for system size.\
                                         11 is minimum.".format(value))
    return value

def check_positive(value):
    """Checks if a given value is positive.

    Args:
        value (any): The value to check.

    Raises:
        argparse.ArgumentTypeError: If the number if not positive.

    Returns:
        float: The value
    """
    value = float(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("{} is an invalid positive value".format(value))
    return value

def check_positive_int(value):
    """Checks if a given value is a positive integer.

    Args:
        value (any): The integer to check.

    Raises:
        argparse.ArgumentTypeError: If the value is not an int or positive.

    Returns:
        int: The value of the integer.
    """
    # try to convert to int
    value = check_positive(value)
    value = int(value)
    return value
