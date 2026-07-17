import logging

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.preprocessing import StandardScaler


def detrend_exponential(y_with_nan):
    """
    Bleach correction via simple exponential fit, subtraction, and re-adding the mean

    Uses np.polyfit on np.log(y), with errors weighted back to the data space. See:

    https://stackoverflow.com/questions/3433486/how-to-do-exponential-and-logarithmic-curve-fitting-in-python-i-found-only-poly

    Parameters
    ----------
    y_with_nan

    Returns
    -------

    """

    ind = np.where(~np.isnan(y_with_nan))[0]
    t = np.squeeze(StandardScaler(copy=False).fit_transform(ind.reshape(-1, 1)))
    y = y_with_nan[ind]
    y_log = np.log(y)

    fit_vars = np.polyfit(t, y_log, 1)#, w=np.sqrt(y))

    # Subtract in the original data space
    y_fit = np.exp(fit_vars[0]) * np.exp(t*fit_vars[1])
    y_corrected = y - y_fit + np.mean(y)

    return ind, y_corrected


def detrend_exponential_lmfit(y_with_nan, x=None, ind_subset=None, restore_mean_value=False, use_const=False):
    """
    Bleach correction via simple exponential fit, division, and (optionally) re-adding the mean

    Uses a direct solver, not a linear regression (does not take the log of the points).

    Parameters
    ----------
    x: optional; time-like vector
    y_with_nan: values to fit
    restore_mean_value
    use_const: whether to add a constant to the fit (extra parameter), or use a raw exponential
        NOTE: if the exponential is very shallow (i.e. almost constant), this term can destabilize the fit

    Returns
    -------

    """
    from lmfit.models import ExponentialModel, ConstantModel
    original_mean = np.nanmean(y_with_nan)

    exp_mod = ExponentialModel(prefix='exp_')
    model = exp_mod
    if use_const:
        const_mod = ConstantModel(prefix='const_')
        model = model + const_mod
    if ind_subset is None:
        ind_subset = y_with_nan.index[np.where(~np.isnan(y_with_nan))[0]]
    if x is None:
        x = ind_subset
    else:
        x = x[ind_subset]
    y = y_with_nan[ind_subset]
    out = None

    try:
        pars = exp_mod.guess(y, x=x)
        if use_const:
            pars += const_mod.guess(y, x=x)
        out = model.fit(y, pars, x=x)

        comps = out.eval_components(x=x)
        y_fit = comps['exp_']
        if use_const:
            y_fit += comps['const_']
        y_corrected = y / y_fit

        y_corrected_with_nan = np.empty_like(y_with_nan)
        y_corrected_with_nan[:] = np.nan
        y_corrected_with_nan = pd.Series(y_corrected_with_nan, index=y_with_nan.index)
        y_corrected_with_nan[ind_subset] = y_corrected
        y_corrected_with_nan = y_corrected_with_nan.values  # Keep legacy behavior
        flag = True

    except (TypeError, ValueError):
        # Occurs when there are too few input points
        logging.warning("Exponential fit failed due to too few points; returning uncorrected values")
        y_corrected_with_nan, y_fit = y_with_nan, y_with_nan
        flag = False

    if out is None or not out.errorbars:# or 0 in y_fit:
        # Crude measurement of bad convergence, even if it didn't error out
        # logging.warning("Exponential fit failed; returning uncorrected values")
        # plt.scatter(x, y, color='gray', label='data')
        # plt.plot(x, y_fit, label='best fit')
        # plt.legend()
        # plt.show()
        y_corrected_with_nan, y_fit = y_with_nan, np.mean(y_with_nan) * np.ones_like(y_with_nan)
        flag = False

    if restore_mean_value:
        y_corrected_with_nan *= original_mean / np.nanmean(y_corrected_with_nan)

    return y_corrected_with_nan, (y_fit, out, flag)


def detrend_exponential_iter(trace, max_iters=100, convergence_threshold=0.01,
                             low_quantile=0.15, high_quantile=0.85, **kwargs):
    """
    Similar to detrend_exponential_lmfit, but it tries to remove outliers and high-activity points

    low/high_quantile: how many percent of the data should be excluded at bottom/top

    Parameters
    ----------
    trace
    convergence_threshold - stop if L2 norm changes by less than this
    low_quantile - per iteration, remove this bottom percentile
    high_quantile - per iteration, remove this bottom percentile

    Returns
    -------

    """
    y_full = trace
    ind_iter = np.where(~np.isnan(y_full))[0]
    y_detrend, num_iter = None, 0

    for num_iter in range(max_iters):
        y_detrend, fit_results = detrend_exponential_lmfit(y_full, ind_subset=ind_iter, **kwargs)
        y_fit = fit_results[0]
        y_fit_last = y_fit

        ind_not_too_small = np.nanquantile(y_detrend, low_quantile) < y_detrend
        ind_not_too_large = y_detrend < np.nanquantile(y_detrend, high_quantile)
        ind_iter = np.where(np.logical_and(ind_not_too_small, ind_not_too_large))[0]
        if scipy.spatial.distance.euclidean(y_fit, y_fit_last) <= convergence_threshold:
            break
    return y_detrend, num_iter


def full_lm_with_windowed_regression_vol(project_data, neuron, window_size=5):
    """" gives back corrected trace for the selected neuron
    calculates alpha for every timepoint considering all neighbours with distance < window size"""

    num_timepoints = project_data.red_traces.shape[0]

    green = np.array(project_data.green_traces[neuron]["intensity_image"])
    vol = np.array(project_data.green_traces[neuron]["area"])
    red = np.array(project_data.red_traces[neuron]["intensity_image"])
    remove_nan = np.logical_and(np.invert(np.isnan(green)), np.invert(np.isnan(vol)))
    green = green[remove_nan]
    vol = vol[remove_nan]
    red = red[remove_nan]

    alpha_green = []
    for i in range(window_size, len(red) - window_size - 1):
        y = green[i - window_size:i + window_size]
        x = vol[i - window_size:i + window_size].reshape(-1, 1)
        model = sklearn.linear_model.LinearRegression(fit_intercept=False)
        model.fit(x, y)
        alpha_green.append(model.coef_)

    alpha_red = []
    for i in range(window_size, len(red) - window_size - 1):
        y = red[window_size:len(red) - window_size - 1]
        x = vol[window_size:len(red) - window_size - 1].reshape(-1, 1)
        model = sklearn.linear_model.LinearRegression(fit_intercept=False)
        model.fit(x, y)
        alpha_red.append(model.coef_)

    red_corrected = red[window_size:len(red) - window_size - 1] - np.array(alpha_red).flatten() * vol[window_size:len(
        red) - window_size - 1]
    green_corrected = green[window_size:len(red) - window_size - 1] - np.array(alpha_green).flatten() * vol[
                                                                                                        window_size:len(
                                                                                                            red) - window_size - 1]

    vol = project_data.red_traces[neuron]["area"][remove_nan][window_size:len(red) - window_size - 1]
    x = project_data.red_traces[neuron]["x"][remove_nan][window_size:len(red) - window_size - 1]
    y = project_data.red_traces[neuron]["y"][remove_nan][window_size:len(red) - window_size - 1]
    z = project_data.red_traces[neuron]["z"][remove_nan][window_size:len(red) - window_size - 1]
    t = np.array(range(num_timepoints))[remove_nan][window_size:len(red) - window_size - 1]
    X = [red_corrected, vol, x, y, z, t]

    X = np.c_[np.array(X).T]
    model = sklearn.linear_model.LinearRegression()
    model.fit(X, green_corrected)
    x_pred = model.predict(X)

    res = green_corrected - x_pred
    return res


def bleach_correct_gaussian_moving_average(y: pd.Series, std=300, subtract=True) -> pd.Series:
    """
    Subtracts or divides traces from a gaussian filtered version of the trace

    See filter_gaussian_moving_average
    """
    y_filt = y.rolling(center=True, window=800, win_type='gaussian', min_periods=1).mean(std=std)
    if subtract:
        return y - y_filt + y_filt.mean()
    else:
        return y / y_filt
