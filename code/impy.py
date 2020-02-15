from itertools import chain
from cvxopt import blas, lapack, solvers
from cvxopt import matrix, spmatrix, sin, mul, div, normal, spdiag
import numpy as np
solvers.options['show_progress'] = 0



def get_second_derivative_matrix(n):
    """
    :param n: The size of the time series

    :return: A matrix D such that if x.size == (n,1), D * x is the second derivate of x
    """
    m = n - 2
    D = spmatrix(list(chain(*[[1, -2, 1]] * m)),
                 list(chain(*[[i] * 3 for i in range(m)])),
                 list(chain(*[[i, i + 1, i + 2] for i in range(m)])))
    return D

def get_first_derivative_matrix(n):
    """
    :param n: The size of the time series

    :return: A matrix D such that if x.size == (n,1), D * x is the second derivate of x
    """
    m = n - 2
    D = spmatrix(list(chain(*[[-1, 1, 0]] * m)),
                 list(chain(*[[i] * 3 for i in range(m)])),
                 list(chain(*[[i, i + 1, i + 2] for i in range(m)])))
    return D



def _l1ctf(corr, delta):  #L1 C


    n = corr.size[0]
    m = n - 2

    D = get_first_derivative_matrix(n)

    P = D * D.T
    q = -D * corr

    G = spmatrix([], [], [], (2*m, m))
    G[:m, :m] = spmatrix(1.0, range(m), range(m))
    G[m:, :m] = -spmatrix(1.0, range(m), range(m))

    h = matrix(delta, (2*m, 1), tc='d')

    res = solvers.qp(P, q, G, h)
    return corr - D.T * res['x']

def _l1tf(corr, delta):    #L1 T

    n = corr.size[0]
    m = n - 2

    D = get_second_derivative_matrix(n)

    P = D * D.T
    q = -D * corr

    G = spmatrix([], [], [], (2*m, m))
    G[:m, :m] = spmatrix(1.0, range(m), range(m))
    G[m:, :m] = -spmatrix(1.0, range(m), range(m))

    h = matrix(delta, (2*m, 1), tc='d')

    res = solvers.qp(P, q, G, h)
    return corr - D.T * res['x']

###########################################################################
def _l1tccf(corr, delta1,delta2):   #L1-TC

    n = corr.size[0]
    m = n - 2

    D1= get_first_derivative_matrix(n)
    D2 = get_second_derivative_matrix(n)
    D=matrix([D1,D2])

    P = D * D.T
    q = -D * corr

    G = spmatrix([], [], [], (4*m, 2*m))
    G[:2*m, :2*m] = spmatrix(1.0, range(2*m), range(2*m))
    G[2*m:, :2*m] = -spmatrix(1.0, range(2*m), range(2*m))
    h1 = matrix(delta1, (2*m, 1), tc='d')
    h2 = matrix(delta2, (2 * m, 1), tc='d')
    h=matrix([h1,h2])
    res = solvers.qp(P, q, G, h)
    return corr - D.T * res['x']


def get_second_derivative_matrix_numpy(n):
    I1 = np.diag([1] * n)
    a1 = np.zeros((n, n + 2))
    for i in range(n):
        a1[i] = np.append(I1[i], [0, 0])

    I2 = np.diag([-2] * n)
    a2 = np.zeros((n, n + 1))
    for i in range(n):
        a2[i] = np.append(0, I2[i])
    a3 = np.zeros((n, n + 2))
    for i in range(n):
        a3[i] = np.append(a2[i], 0)

    a4 = np.zeros((n, n + 1))
    for i in range(n):
        a4[i] = np.append(0, I1[i])
    a5 = np.zeros((n, n + 2))
    for i in range(n):
        a5[i] = np.append(0, a4[i])

    return a1 + a3 + a5


def _hp(corr, delta):   #HP


    n = corr.size[0]
    m = n - 2

    D = get_second_derivative_matrix_numpy(m)

    P = np.dot(D.T,D)
    I=np.diag([1]*(int(np.sqrt(P.size))))
    mat=I+2*delta*P
    mat_=np.linalg.inv(mat)
    mat_2=np.array(np.dot(mat_,corr))
    mat_2=mat_2.flatten();

    #print(mat_2.size)

    return mat_2
