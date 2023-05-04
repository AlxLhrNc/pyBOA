# -*- coding: utf-8 -*-
"""
Created on Fri May 13 11:31:52 2022

@author: AlxLhrNC
"""

# Packages

import numpy as np
import xarray as xr
import pandas as pd
from math import pi, floor, ceil
from scipy.ndimage import sobel, correlate
from skimage import morphology
from scipy.stats import norm


types = xr.core.dataset.Dataset, xr.core.dataarray.DataArray

# %% spur removal
"""
The spur removal section was implemented by Ben Mabey: https://gist.github.com/bmabey, see bwmorph
"""

LUT_DEL_MASK = np.array([[8, 4, 2], [16, 0, 1], [32, 64, 128]], dtype=np.uint8)


def _bwmorph_luts(image, luts, n_iter=None, padding=0):
    # check parameters
    if n_iter is None:
        n = -1
    elif n_iter <= 0:
        raise ValueError("n_iter must be > 0")
    else:
        n = n_iter
    # check that we have a 2d binary image, and convert it
    # to uint8
    im = np.array(image).astype(np.uint8)

    if im.ndim != 2:
        raise ValueError("2D array required")
    if not np.all(np.in1d(image.flat, (0, 1))):
        raise ValueError("Image contains values other than 0 and 1")
    # iterate either 1) indefinitely or 2) up to iteration limit
    while n != 0:
        before = np.sum(im)  # count points before

        # for each subiteration
        for lut in luts:
            # correlate image with neighborhood mask
            N = correlate(im, LUT_DEL_MASK, mode="constant", cval=padding)
            # take deletion decision from this subiteration's LUT
            D = np.take(lut, N)
            # perform deletion
            im[D] = 0
        after = np.sum(im)  # count points after

        if before == after:
            # iteration had no effect: finish
            break
        # count down to iteration limit (or endlessly negative)
        n -= 1
    return im.astype(bool)


SPUR_LUT = np.array([1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.bool)


def spur(image, n_iter=None):
    """
    Removes "spurs" from an image

    Parameters
    ----------
    image : binary (M, N) ndarray
        The image to be spurred.

    n_iter : int, number of iterations, optional
        Regardless of the value of this parameter, the de-spurred image
        is returned immediately if an iteration produces no change.
        If this parameter is specified it thus sets an upper bound on
        the number of iterations performed.

    Returns
    -------
    out : ndarray of bools
        de-spurred image.


    Examples

    --------
  >>> t = np.array([[0, 0, 0, 0],
                     [0, 0, 1, 0],
                     [0, 1, 0, 0],
                     [1, 1, 0, 0]])
  >>> spur(t).astype(np.uint8)
      array([[0 0 0 0]
             [0 0 0 0]
             [0 1 0 0]
             [1 1 0 0]]
    """
    return _bwmorph_luts(image, [SPUR_LUT], n_iter=n_iter, padding=1)


# %% BOAarray
def BOAarray(array, dims=None):
    """
    Standardize the array type (xarray DataArray/DataSet) and dimensions names.

    Parameters
    ----------
    array : numpy/pandas/xarray arrays
        array to standardize.
    dims : list, optional
        List of str to use for the xarray dimensions. The default is ["time", "lat", "lon"].

    Raises
    ------
    TypeError
        As of now, only deal with xarray, pandas, numpy arrays.

    Returns
    -------
    array : xarray DataArray (DataSet if input was DataSet)
        Stardardized array.

    """
    if isinstance(array, (xr.core.dataarray.DataArray, xr.core.dataset.Dataset)):
        if dims is None:
            dims = ["time", "lat", "lon"]
        names_dict = {key: value for (key, value) in zip(array.dims, dims)}
        array = array.rename(names_dict)
    elif isinstance(array, np.ndarray):
        array = np.expand_dims(array.squeeze(), axis=0)
        array = xr.DataArray(array, dims=dims)
    elif isinstance(array, pd.core.frame.DataFrame):
        array = xr.DataArray(array, dims=dims[1:]).expand_dims(dim=dims[0])
    else:
        raise TypeError(
            f"{type(array)} not supported. Switch to pandas, numpy or xarray arrays."
        )
    return array


# %% pyBOA class


@xr.register_dataarray_accessor("pyBOA")
@xr.register_dataset_accessor("pyBOA")
class pyBOA:
    def __init__(self, array):
        """
        xarray accessor for pyBOA.
        Use as:
            xarray.DataArray.pyBOA()

        Parameters
        ----------
        array : xarray DataArray / DataSet
            xarray DataArray / DataSet. Use BOAarray function if unsure.

        Returns
        -------
        None.

        """
        self._array = array

    # %% flag_n
    def flag_n(self, n):
        """
        Flags extremas in n*n window

        Parameters
        ----------
        n : int
            Size of the window.

        Returns
        -------
        flag : xarray DataArray/DataSet
            Boolean array with extremas as TRUE.

        """
        array = self._array
        window_size = {name: n for name in ["lat", "lon"]}
        window = array.rolling(window_size, center=True)

        peak_min = window.min(skipna=True)
        peak_max = window.max(skipna=True)

        flag = (peak_min == array) | (peak_max == array)

        return flag

    # %% mfMinN
    def mfNinM(self, m=5, n=3, return_filter=False):
        """
        Contextual Median Filtering

        Parameters
        ----------
        m : int, optional
            Size of the big window. The default is 5.
        n : int, optional
            Size of the small window. The default is 3.
        return_filter : bool, optional
            Do you want the filter to be returned by the function. The default is False.

        Raises
        ------
        ValueError
            `m` being the size of the big window, it should always be smaller than `n` .

        Returns
        -------
        xarray DataArray/DataSet (both)
            filtered_nc the filtered version of the array.
            to_filter the mask that was filtered.

        WARNING: As per BOA design, works with 2 implementation of flag_n with m = 5 and n = 3.
        """
        if m <= n:
            raise ValueError(
                "Large window m:{m} can not be equal or smaller than small window n:{n}"
            )
        array = self._array
        if len(array.dims) == 2:
            array = array.expand_dims({"tmp": 1})
        Ninf, Nsup = floor(n / 2), ceil(n / 2)
        peak_M = array.pyBOA.flag_n(m)
        peak_N = array.pyBOA.flag_n(n)
        to_filter = peak_N * ~peak_M
        filtered_nc = array.copy()
        idx = np.where(to_filter)
        for it, ix, iy in zip(*idx):
            window = array[it, ix - Ninf : ix + Nsup, iy - Ninf : iy + Nsup]
            filtered_nc[it, ix, iy] = window.median(skipna=True)
        if return_filter:
            return filtered_nc, to_filter
        else:
            return filtered_nc

    # %% sobel_haversine
    def sobel_haversine(self):
        """
        Adjusted version of the sobel gradient to take into account the distance distortion.

        Returns
        -------
        sobel_grd : xarray DataArray/DataSet
            The sobel gradient adjusted with the haversine formula.

        """
        array = self._array
        dimensions = array.dims
        coordinates = array.coords

        _, hvrsn = np.meshgrid(
            array.lon, np.cos(array.lat * pi / 180)
        )  # extracting cos(lat) as a matrix

        sobel_hzt, sobel_vrt = (
            sobel(array, 1),
            sobel(array, 2),
        )  # Sobel along the longitude

        # gradient calculation
        sobel_grd = hvrsn * np.hypot(sobel_hzt, sobel_vrt)
        sobel_grd = xr.DataArray(sobel_grd, coords=coordinates, dims=dimensions)
        return sobel_grd

    # %% front_trsh
    def front_trsh(self, wndw=64, prcnt=90):
        """
        Fronts thresholding using percentile.

        Parameters
        ----------
        wndw : int, optional
            Window size. The default is 64.
        prcnt : int, optional
            Percentile. The default is 90.

        Returns
        -------
        frnt : xarray DataArray/DataSet
            Boolean array TRUE where the values are above the chosen percentile.

        """
        array = self._array
        window = array.rolling(
            {name: wndw for name in ["lat", "lon"]}, center=True, min_periods=1
        )
        window_qt = window.reduce(np.nanpercentile, q=prcnt)
        frnt = self._array > window_qt
        return frnt

    # %% thinning
    def thinning(self, iteration=2, min_size=7, f_dilate=False):
        """
        Used on the output from `pyBOA.front_trsh()` to obtain single lines.

        Parameters
        ----------
        iteration : int, optional
            Number of cleaning loop to execute. The default is 2.
        min_size : int, optional
            Minimal size of the structures to remove. The default is 7.
        f_dilate : bool, optional
            Should the function make a first dilation before cleaning loops. The default is False.

        Returns
        -------
        frnt : xarray DataArray/DataSet
            Boolean array TRUE where a front was detected.

        """
        array = self._array
        frnt = array.to_numpy().squeeze()

        if f_dilate:
            # dilate
            frnt = morphology.dilation(frnt)
        for it in range(iteration):
            # morphological thining
            frnt = morphology.thin(frnt)
            # spur removal
            frnt[frnt > 1] = 1
            frnt = spur(frnt, n_iter=1)
            # clean small object
            frnt = morphology.remove_small_objects(
                frnt.astype(bool), min_size=min_size, connectivity=2
            )
            # remove small holes
            frnt = morphology.remove_small_holes(frnt)
            if it < iteration - 1:
                # dilate
                frnt = morphology.dilation(frnt)
        frnt = morphology.thin(frnt)
        if len(array.dims) == 3:
            frnt = np.expand_dims(frnt, axis=0)
        frnt = xr.DataArray(frnt, dims=array.dims, coords=array.coords)

        return frnt

    # %% roll_conf_int
    def roll_conf_int(self, wndw=64, ci=0.75):
        """
        rolling confidence interval

        Parameters
        ----------
        wndw : int, optional
            Window size. The default is 64.
        ci : float, optional
            confidence interval within [0,1]. The default is 0.75.

        Returns
        -------
        array : xarray DataArray/DataSet
            array with values out of the confidence interval

        """
        array = self._array.where(np.isfinite(self._array.values))
        window = array.rolling(
            {name: wndw for name in ["lat", "lon"]}, center=True, min_periods=1
        )
        mean_ = window.reduce(np.nanmean)
        sd_ = window.reduce(np.nanstd)
        vmin, vmax = norm.interval(ci, loc=mean_, scale=sd_)
        array = array.where((array < vmin) | (array > vmax))

        return array


# %% END
