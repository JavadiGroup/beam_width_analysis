import pandas as pd
import numpy as np
import scipy.special as special
import scipy.optimize as optimize
from typing import List, Tuple

def guassian_beam_erfc(x: pd.Series, p_offset: float, p_max: float, x_half: float, w: float) -> pd.Series:
    """
    Guassian beam model function for use with knife edge profiling method. uses an erfc function to fit data. 

    This function models the intensity profile of a guassian beam and can be used to calculate beam width when fitted to data.

    ensure that the units for the data used to fit are of the same degree of magnitude. For example (mW, mm) or (W, m) etc.

    Args:
        x: Input data points where the model is evaluated (x axis).
        p_offset: backgroound noise (room lights, laser noise, etc).
        p_max: maximum beam intensity.
        x_half: position of the beam intensity half way between the maximum and the offset.
        w: beam radius.

    Returns:
        pd.Series: the intensity of the beam at each x value.
    """
    y = (x - x_half)/(w/np.sqrt(2))
    return p_offset + ((p_max/2) * special.erfc(y))


def fit_data(model_function: callable, x_data: pd.Series, y_data: pd.Series, initial_guess: List[float]) -> List[np.ndarray]:
    """
    Fits the provided model function to the data and returns the parameters of the fit and their uncertainties.

    Args:
        model_function: The model function to fit the data to.
        x_data: the x data to fit the model to.
        y_data: the y data to fit the model to.
        initial_guess: the initial guess for the parameters of the model.

    Returns:
        List[np.ndarray]: the parameters of the fit and the covariance matrix. 
    """
    param, covariance = optimize.curve_fit(model_function, x_data, y_data, p0=initial_guess)
    return [param, covariance]


