# -*- coding: utf-8 -*-
"""
Linear algebra-related functions.

@author: Rui Yang
"""

import numpy as np
from fractions import Fraction
from MathSpace.base import log

def solve_lineqn(x, y, as_frac = True, to_float = True, data_type = None):
    """Using the Gauss-Jordan reduction method to solve the linear equation systems.
    
    x: should be a numerical and 2-dimensional list or numpy array object.
    
    y: should be a numerical and 1-dimensional list or numpy array object with the same number of elements as the number of rows of `x`. Internally, `x` and `y` will be combined into an augmented matrix.
    
    as_frac: whether to represent values in `x` as fractions.
    
    to_float: convert the fraction values to the float values after calculation if `as_frac = True`.
    
    data_type: a valid numpy data type (e.g., `numpy.float64` which is the default if `data_type` is `None` and `as_frac` is `False`).
    
    Note: we always recommend using `as_frac = True`. If not, using `data_type = numpy.float128` is strongly recommended. Otherwise; error from the floating-point arithmetic may cause absolutely wrong results."""
    as_frac_func = np.vectorize(Fraction)
    
    if not isinstance(y, (list, np.ndarray)):
        raise TypeError('y can only be a list or numpy array object!')
    
    y = np.array(y, dtype = np.float64 if not data_type else data_type) if isinstance(y, list) else y
    
    if not (y.ndim == 1):
        raise TypeError('y must only have one dimension!')
    
    x = gj_reduc(x, y, as_frac = as_frac, to_float = to_float, data_type = data_type)
    
    x_copy = x.copy()
    x = x['x_y']
    
    # Remove rows whose values all are zeros
    x = x[~((x != 0).sum(axis = 1) == 0)]
    
    # If x contains row(s) like [0 0 ... 0|1], it has no solution
    if (((x[:, :-1] != 0).sum(axis = 1) == 0) & (x[:, -1] != 0)).any():
        return {'solution_flag': 'NO', 'x': x_copy}
    
    # If x has a strict triangular form, it has an unique solution
    identity_matrix = as_frac_func(np.identity(x.shape[0])) if as_frac else np.identity(x.shape[0])
    if (x.shape[0] == x.shape[1] - 1) and (x[:, :-1] == identity_matrix).all():
            return {'solution_flag': 'UNIQUE', 'x': x_copy, 'solution': x[:, -1]}
    
    # If x contains free variable(s), it has infinitely many solutions
    solution_dict = {}
    for row_index in range(0, x.shape[0], 1):
        non_zero_flag = False
        for i in range(0, x.shape[1] - 1, 1):
            if (not non_zero_flag) and (x[row_index][i] != 0):
                non_zero_flag = True
                solution_key = num2unknown(x[row_index][i], i + 1, to_float = to_float, no_plus_sign = True)
                solution_dict[solution_key] = '' if x[row_index][-1] == 0 else str(x[row_index][-1])
            elif non_zero_flag and (x[row_index][i] != 0):
                solution_dict[num2unknown(1, i + 1, to_float = to_float, no_plus_sign = True)] = num2unknown(1, i + 1, to_float = to_float, no_plus_sign = True)
                solution_dict[solution_key] += num2unknown(-x[row_index][i], i + 1)
        solution_dict[solution_key] = solution_dict[solution_key].strip('+')
    return {'solution_flag': 'INFINITE', 'x': x_copy, 'solution': solution_dict}

def inv(x, as_frac = True, to_float = True, data_type = None):
    """Get the inverse matrix if available using (A|I) --> (I|A^-1).
    
    x: must be a square matrix.
    
    as_frac: whether to represent values in `x` as fractions.
    
    to_float: convert the fraction values to the float values after calculation if `as_frac = True`.
    
    data_type: a valid numpy data type (e.g., `numpy.float64` which is the default if `data_type` is `None` and `as_frac` is `False`).
    
    Note: we always recommend using `as_frac = True`. If not, using `data_type = numpy.float128` is strongly recommended. Otherwise; error from the floating-point arithmetic may cause absolutely wrong results."""
    if not isinstance(x, (list, np.ndarray)):
        raise TypeError('x can only be a list or numpy array object!')
    
    x = np.array(x, dtype = np.float64 if not data_type else data_type) if isinstance(x, list) else x
    
    if not (x.ndim == 2 and (np.array(x.shape) >= 2).all() and x.shape[0] == x.shape[1]):
        raise TypeError('x must be a 2-dimensional square matrix whose sizes >= 2 exactly!')
    
    y = np.identity(x.shape[0], dtype = np.float64 if not data_type else data_type)
    
    x = gj_reduc(x, y, as_frac = as_frac, to_float = to_float, data_type = data_type)
    
    if (x['x_new'] == x['y']).all():
        x['solution_flag'] = 'NONSINGULAR'
    else:
        x['solution_flag'] = 'SINGULAR'
    
    return x
    
def gj_reduc(x, y, as_frac = True, to_float = True, data_type = None):
    """Using the Gauss-Jordan reduction to convert a matrix to a reduced row echelon form.
    
    x: should be a numerical and 2-dimensional list or numpy array object.
    
    y: should be a numerical list or numpy array object with the same number of rows as `x`. Internally, `x` and `y` will be combined into an augmented matrix.
    
    as_frac: whether to represent values in `x` as fractions.
    
    to_float: convert the fraction values to the float values after calculation if `as_frac = True`.
    
    data_type: a valid numpy data type (e.g., `numpy.float64` which is the default if `data_type` is `None` and `as_frac` is `False`).
    
    Note: we always recommend using `as_frac = True`. If not, using `data_type = numpy.float128` is strongly recommended. Otherwise; error from the floating-point arithmetic may cause absolutely wrong results."""
    to_float_func = np.vectorize(float)
    as_frac_func = np.vectorize(Fraction)
    
    if not isinstance(x, (list, np.ndarray)):
        raise TypeError('x can only be a list or numpy array object!')
    
    if not isinstance(y, (list, np.ndarray)):
        raise TypeError('y can only be a list or numpy array object!')
    
    x = np.array(x, dtype = np.float64 if not data_type else data_type) if isinstance(x, list) else x
    
    y = np.array(y, dtype = np.float64 if not data_type else data_type) if isinstance(y, list) else y
    
    if not (x.ndim == 2 and (np.array(x.shape) >= 2).all()):
        raise TypeError('x must have 2 dimensions whose sizes >= 2 exactly!')
    
    if not (y.ndim <= 2 and y.shape[0] == x.shape[0]):
        raise TypeError('y must have at most 2 dimensions and have the same number of rows as x!')
    
    x_copy = x.copy()
    x = np.concatenate((x, y if y.ndim == x.ndim else y.reshape((y.shape[0], 1))), axis = 1)
    
    if as_frac and (x.dtype != np.float128):
        log.info('Fraction mode enabled!')
        x = as_frac_func(x)
    
    # Forward elimination (let every entry below each pivot be 0)
    pivotal_row, pivotal_col = 0, 0
    while pivotal_row < x.shape[0] - 1 and pivotal_col < x.shape[1] - 1:
        if np.abs(x[pivotal_row:x.shape[0], pivotal_col]).max() > 0:
            x[[pivotal_row, np.abs(x[pivotal_row:x.shape[0], pivotal_col]).argmax() + pivotal_row]] = x[[np.abs(x[pivotal_row:x.shape[0], pivotal_col]).argmax() + pivotal_row, pivotal_row]]
            for row_index in range(pivotal_row + 1, x.shape[0], 1):
                if x[row_index, pivotal_col] != 0:
                    if as_frac:
                        x[row_index, :] = x[row_index, :] + x[pivotal_row, :] * Fraction(-x[row_index, pivotal_col], x[pivotal_row, pivotal_col])
                    else:
                        x[row_index, :] = x[row_index, :] + x[pivotal_row, :] * -x[row_index, pivotal_col] / x[pivotal_row, pivotal_col]
            pivotal_row, pivotal_col = pivotal_row + 1, pivotal_col + 1
        else:
            pivotal_col += 1
    
    # Let every lead variable be 1
    for row_index in range(0, x.shape[0], 1):
        if (x[row_index, (np.abs(x[row_index, :x.shape[1] - 1]) > 0).argmax()] != 0) and (x[row_index, (np.abs(x[row_index, :x.shape[1] - 1]) > 0).argmax()] != 1):
            if as_frac:
                x[row_index, :] = x[row_index, :] / Fraction(x[row_index, (np.abs(x[row_index, :x.shape[1] - 1]) > 0).argmax()])
            else:
                x[row_index, :] = x[row_index, :] / x[row_index, (np.abs(x[row_index, :x.shape[1] - 1]) > 0).argmax()]
    
    # Backward elimination (let every lead variable be the uniquely non-zero entry in the column in which it locates)
    for lead_row_index in range(x.shape[0] - 1, 0, -1):
        if (np.abs(x[lead_row_index, :x.shape[1] - 1]) > 0).any():
            for forward_row_index in range(lead_row_index - 1, -1, -1):
                if x[forward_row_index, (np.abs(x[lead_row_index, :x.shape[1] - 1]) > 0).argmax()] != 0:
                    if as_frac:
                        x[forward_row_index, :] = x[forward_row_index, :] + x[lead_row_index, :] * Fraction(-x[forward_row_index, (np.abs(x[lead_row_index, :x.shape[1] - 1]) > 0).argmax()], x[lead_row_index, (np.abs(x[lead_row_index, :x.shape[1] - 1]) > 0).argmax()])
                    else:
                        x[forward_row_index, :] = x[forward_row_index, :] + x[lead_row_index, :] * -x[forward_row_index, (np.abs(x[lead_row_index, :x.shape[1] - 1]) > 0).argmax()] / x[lead_row_index, (np.abs(x[lead_row_index, :x.shape[1] - 1]) > 0).argmax()]
    
    return {'x_y': to_float_func(x) if to_float and as_frac else x,
            'x_new': to_float_func(x[:, :x_copy.shape[1]]) if to_float and as_frac else x[:, :x_copy.shape[1]],
            'y_new': to_float_func(x[:, x_copy.shape[1]:]) if to_float and as_frac else x[:, x_copy.shape[1]:],
            'x': x_copy, 'y': y}

def num2unknown(x, rank, to_float = False, no_plus_sign = False):
    """Formatting `x` to unknown.
    
    x: the coefficient of the unknown.
    
    rank: the rank of the unknown.
    
    to_float: convert x to a floating-point value.
    
    no_plus_sign: does not add "+" before `x`.
    
    e.g., return "-10x9" if `x = -10` and `rank = 9`."""
    x = float(x) if to_float else x
    if x == 0:
        return ''
    elif x == 1 and no_plus_sign:
        return 'x' + str(rank)
    elif x == 1 and (not no_plus_sign):
        return '+x' + str(rank)
    elif x == -1:
        return '-x' + str(rank)
    elif x > 0 and no_plus_sign:
        return str(x) + 'x' + str(rank)
    elif x > 0 and (not no_plus_sign):
        return '+' + str(x) + 'x' + str(rank)
    elif x < 0:
        return str(x) + 'x' + str(rank)