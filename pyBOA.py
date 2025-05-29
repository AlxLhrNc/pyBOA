# -*- coding: utf-8 -*-
"""
Created on Fri May 13 11:31:52 2022

@author: AlxndrLhrNc
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
The spur removal section was implemented by Ben Mabey: https://gist.github.com/bmabey
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
    if not np.all(np.isin(image.flat, (0, 1))): # formerly np.all(np.in1d(image.flat, (0, 1)))
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


SPUR_LUT = np.array(
    [
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ],
    dtype=bool,
)


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
def BOAarray(array, dims: list = ["time", "latitude", "longitude"]):
    if isinstance(array, (xr.core.dataarray.DataArray, xr.core.dataset.Dataset)):
        n = 1
        while len(array.dims) < 3:
            array = array.expand_dims({f'tmp_{n}': 1})
            n += 1
        names_dict = {key: value for (key, value) in zip(array.dims, dims)}
        array = array.rename(names_dict)
    elif isinstance(array, np.ndarray):
        while len(array.shape) < 3:
            array = np.expand_dims(array, 0)
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
        self._array = BOAarray(array)
        self._buffer_ftprnt = morphology.disk(4, dtype=np.float32)

    def __buffer__(self, array):
        """
        generate the buffer needed for auto-detection

        Parameters
        ----------
        array : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return morphology.binary_dilation(
            np.isnan(array).squeeze(), footprint=self._buffer_ftprnt)

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
        window_size = {name: n for name in ["latitude", "longitude"]}
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

        WARNING: As per BOA design, works with 2 implementation of flag_n with m = 5 and n = 3 by default.
        """
        if m <= n:
            raise ValueError(
                "Large window m:{m} can not be equal or smaller than small window n:{n}"
            )
        array = self._array
        Ninf, Nsup = floor(n / 2), ceil(n / 2)
        peak_M = array.pyBOA.flag_n(m)
        peak_N = array.pyBOA.flag_n(n)
        to_filter = peak_N * ~peak_M
        filtered_nc = array.copy()
        idx = np.where(to_filter)
        for it, ix, iy in zip(*idx):
            window = array[it, ix - Ninf: ix + Nsup, iy - Ninf: iy + Nsup]
            filtered_nc[it, ix, iy] = window.median(skipna=True)

        return (filtered_nc, to_filter) if return_filter else filtered_nc

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

        _, hvrsn = np.meshgrid(
            array.longitude, np.cos(array.latitude * pi / 180)
        )  # extracting cos(lat) as a matrix

        sobel_hzt, sobel_vrt = (
            sobel(array, 1),
            sobel(array, 2),
        )  # Sobel along the longitude

        # gradient calculation
        sobel_grd = hvrsn * np.hypot(sobel_hzt, sobel_vrt)
        sobel_grd = xr.DataArray(
            sobel_grd, coords=array.coords, dims=array.dims)
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
            {name: wndw for name in ["latitude", "longitude"]}, center=True, min_periods=1
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

        for t in range(len(array.time)):
            frnt = array[t, :, :].squeeze()  # make it work with .sel(time=t)
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
            array[t, :, :] = frnt  # .sel(time=t)

        return array

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
            {name: wndw for name in ["latitude", "longitude"]}, center=True, min_periods=1
        )
        mean_ = window.reduce(np.nanmean)
        sd_ = window.reduce(np.nanstd)
        vmin, vmax = norm.interval(ci, loc=mean_, scale=sd_)
        array = array.where((array < vmin) | (array > vmax))

        return array

    # %% roll_percent
    def roll_percent(self, wndw=64, prcnt=75):
        """
        rolling confidence interval

        Parameters
        ----------
        wndw : int, optional
            Window size. The default is 64.
        ci : float, optional
            confidence interval within [0,100]. The default is 75.

        Returns
        -------
        array : xarray DataArray/DataSet
            array with values out of the confidence interval

        """
        array = self._array.where(np.isfinite(self._array.values))
        window = array.rolling(
            {name: wndw for name in ["latitude", "longitude"]}, center=True, min_periods=1
        )
        qs = [(100 - prcnt) / 2, 100 - (100 - prcnt) / 2]
        vmin, vmax = (
            window.reduce(np.nanpercentile, q=qs[0]),
            window.reduce(np.nanpercentile, q=qs[1]),
        )
        array = array.where((array < vmin) | (array > vmax))

        return array

    # %% auto_detection
    def auto_detection(self, rmse_target: float = 0.01):
        if isinstance(self._array, xr.core.dataarray.DataArray):
            self._array = self._array.to_dataset()
        for v in self._array.data_vars:
            rmse = 1
            array_copy = self._array[v].copy()
            while rmse > rmse_target:
                res_fltrd = array_copy.pyBOA.mfNinM(return_filter=False)
                # differences projected vs measures
                delta_nc = np.subtract(res_fltrd, array_copy)
                rmse = np.sqrt(
                    np.nansum(delta_nc ** 2) / np.nansum(delta_nc / delta_nc)
                )
                array_copy = res_fltrd.copy()
            res_sobel = res_fltrd.where(~self.__buffer__(
                self._array[v])).pyBOA.sobel_haversine()
            res_frnt = res_sobel.pyBOA.front_trsh().pyBOA.thinning(f_dilate=True)

            array_copy[f"{v}_filtered"] = (array_copy.dims, res_fltrd.data)
            array_copy[f"{v}_sobel"] = (array_copy.dims, res_sobel.data)
            array_copy[f"{v}_fronts"] = (
                array_copy.dims, res_frnt.where(res_frnt > 0).data)

        return array_copy


# %% __main__
if __name__ == "__main__":
    import os
    from cartopy import crs as ccrs, feature as cfeature
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    from matplotlib import pyplot as plt, ticker as mticker
    from matplotlib.colors import LogNorm
    from warnings import simplefilter
    from numpy import array

    simplefilter("ignore", category=RuntimeWarning) # avoid RuntimeWarning: invalid value encountered in log and all NaN slices
    print(f"\nRunning {os.path.basename(__file__)} ...")
    PATH_ROOT = os.path.dirname(os.path.abspath(__file__))
    # Downloading fresh file from Copernicus ############################################################
    minimum_latitude, maximum_latitude, minimum_longitude, maximum_longitude = -37.24822666266156, -35.664520, 174.4191057667842, 176.18599269342826
    projection = ccrs.PlateCarree(central_longitude=180)
    nc_file = os.path.join(PATH_ROOT, "chl_l3_rep_300m_2022_03_15.nc")
    
    # pyBOA ############################################################
    ds = xr.open_dataset(nc_file)
    boa = ds["CHL"].pyBOA.auto_detection()
    print("Test passed on dataarray.")
    # boa2 = ds.pyBOA.auto_detection()
    # print("Test passed on dataset.")
    
    # Visuals ############################################################
    # Base
    [minimum_longitude, minimum_latitude, _], [maximum_longitude, maximum_latitude, _] = projection.transform_points(
        ccrs.PlateCarree(), 
        array([minimum_longitude, maximum_longitude]), 
        array([minimum_latitude, maximum_latitude])
        )
    fig, mppng = plt.subplots(subplot_kw=dict(projection=projection))
    mppng.set_extent([minimum_longitude, maximum_longitude, minimum_latitude, maximum_latitude], crs=projection)
    mppng.add_feature(
            cfeature.GSHHSFeature(scale="auto"),
            facecolor="lightgray",
            edgecolor="black",
            linewidth=0.5,
            zorder=10,
        )
    grd = mppng.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True,
            linewidth=1,
            color="black",
            alpha=0.5,
            linestyle="dotted",
            zorder=20,
        )
    grd.top_labels = False
    grd.right_labels = False
    grd.xformatter = LONGITUDE_FORMATTER
    grd.yformatter = LATITUDE_FORMATTER
    grd.xlocator = mticker.MultipleLocator(0.5)
    grd.ylocator = mticker.MultipleLocator(0.5)
    
    raster_1 = mppng.pcolormesh(boa.longitude,
                              boa.latitude,
                              boa.squeeze(),
                              cmap="viridis",
                              norm=LogNorm(vmin=np.exp(-1), vmax=np.exp(1)),
                              transform=ccrs.PlateCarree(),
                              )
    clrbr = fig.colorbar(
        raster_1, ax=mppng, aspect=30, pad=0.02, shrink=0.6
    )  # pad=distance to figure
    clrbr.set_label("Chlorophyll concentration [mg.m-3]", fontsize=9)
    clrbr.set_ticks([0.4, 0.6, 1, 2])
    clrbr.set_ticklabels([0.4, 0.6, 1, 2])
    mppng.pcolormesh(
        boa.longitude,
        boa.latitude,
        boa["CHL_fronts"].squeeze(),
        cmap="Reds",
        vmin=0,
        vmax=1.3,
        transform=ccrs.PlateCarree(),
    )
    fig.savefig(
        os.path.join(PATH_ROOT, "pyBOA_results.png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.1,
    )
    # END ############################################################
    print(f"\nEnd of the program {os.path.basename(__file__)}", "\a")
