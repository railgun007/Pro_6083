from statsmodels.robust.scale import mad
from cvxopt import matrix, spmatrix, sin, mul, div, normal, spdiag
import pandas as pd
from impl import _l1tf,_hp,_l1ctf,_l1tccf
import numpy as np

def l1tf(corr, delta):   #L1 T
    """
    :param corr: Corrupted signal, should be a numpy array / pandas Series
    :param delta: Strength of regularization

    :return: The filtered series
    """

    m = float(corr.min())
    M = float(corr.max())
    denom = M - m
    # if denom == 0, corr is constant
    t = (corr-m) / (1 if denom == 0 else denom)

    if isinstance(corr, np.ndarray):
        values = matrix(t)
    elif isinstance(corr, pd.Series):
        values = matrix(t.values[:])
    else:
        raise ValueError("Wrong type for corr")

    values = _l1tf(values, delta)
    values = values * (M - m) + m

    if isinstance(corr, np.ndarray):
        values = np.asarray(values).squeeze()
    elif isinstance(corr, pd.Series):
        values = pd.Series(values, index=corr.index, name=corr.name)

    return values

def l1ctf(corr, delta):#L1 C
    """
    :param corr: Corrupted signal, should be a numpy array / pandas Series
    :param delta: Strength of regularization

    :return: The filtered series
    """

    m = float(corr.min())
    M = float(corr.max())
    denom = M - m
    # if denom == 0, corr is constant
    t = (corr-m) / (1 if denom == 0 else denom)

    if isinstance(corr, np.ndarray):
        values = matrix(t)
    elif isinstance(corr, pd.Series):
        values = matrix(t.values[:])
    else:
        raise ValueError("Wrong type for corr")

    values = _l1ctf(values, delta)
    values = values * (M - m) + m

    if isinstance(corr, np.ndarray):
        values = np.asarray(values).squeeze()
    elif isinstance(corr, pd.Series):
        values = pd.Series(values, index=corr.index, name=corr.name)

    return values

def l1tccf(corr, delta1,delta2):  #L1 TC
    """
    :param corr: Corrupted signal, should be a numpy array / pandas Series
    :param delta: Strength of regularization

    :return: The filtered series
    """

    m = float(corr.min())
    M = float(corr.max())
    denom = M - m
    # if denom == 0, corr is constant
    t = (corr-m) / (1 if denom == 0 else denom)

    if isinstance(corr, np.ndarray):
        values = matrix(t)
    elif isinstance(corr, pd.Series):
        values = matrix(t.values[:])
    else:
        raise ValueError("Wrong type for corr")

    values = _l1tccf(values, delta1,delta2)
    values = values * (M - m) + m

    if isinstance(corr, np.ndarray):
        values = np.asarray(values).squeeze()
    elif isinstance(corr, pd.Series):
        values = pd.Series(values, index=corr.index, name=corr.name)

    return values







def hp(corr, delta): #HP
    """
    :param corr: Corrupted signal, should be a numpy array / pandas Series
    :param delta: Strength of regularization

    :return: The filtered series
    """

    m = float(corr.min())
    M = float(corr.max())
    denom = M - m
    # if denom == 0, corr is constant
    t = (corr - m) / (1 if denom == 0 else denom)

    if isinstance(corr, np.ndarray):
        values = matrix(t)
    elif isinstance(corr, pd.Series):
        values = matrix(t.values[:])
    else:
        raise ValueError("Wrong type for corr")

    values = _hp(values, delta)
    values=values*(M-m)+m

    if isinstance(corr, np.ndarray):
        values = np.asarray(values).squeeze()
    elif isinstance(corr, pd.Series):
        values = pd.Series(values, index=corr.index, name=corr.name)

    return values