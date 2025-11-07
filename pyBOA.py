# -*- coding: utf-8 -*-
"""
Created on Fri May 13 11:31:52 2022

@author: AlxndrLhrNc
"""

# Packages

import logging
import numpy as np
import pandas as pd
from scipy.ndimage import sobel, correlate, median_filter
from scipy.stats import norm
from skimage import morphology
from typing import List, Union
import xarray as xr

#%% OPTIONAL LIBRARIES ############################################################
try:
    import dask.array as da  # type: ignore
    _HAS_DASK = True
except Exception:
    _HAS_DASK = False

#%% COMMONS (keep in case) ############################################################
types = (xr.core.dataset.Dataset, xr.core.dataarray.DataArray)

# %% spur removal ############################################################
LUT_DEL_MASK = np.array([[8, 4, 2], [16, 0, 1], [32, 64, 128]], dtype=np.uint8)

def _bwmorph_luts(image, luts, n_iter=None, padding=0):
    """
    Perform binary morphological operations on an image using lookup tables (LUTs).
    INFO: implemented by Ben Mabey: https://gist.github.com/bmabey

    Args:
        image (2D array-like): The binary image to be processed.
        luts (list of 1D arrays): List of lookup tables for deletion decisions.
        n_iter (int, optional): Number of iterations to perform. If None, iterates indefinitely.
        padding (int, optional): Padding value for the correlation operation.
    Returns:
        ndarray: The processed binary image after applying the morphological operations.
    Raises:
        ValueError: If n_iter is not positive or if the image is not a 2D binary array.
    """
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
    INFO: implemented by Ben Mabey: https://gist.github.com/bmabey

    Args:
        image (binary M*N ndarray): the image to be spurred.

        n_iter (int, number of iterations, optional):
            Regardless of the value of this parameter, the de-spurred image
            is returned immediately if an iteration produces no change.
            If this parameter is specified it thus sets an upper bound on
            the number of iterations performed.

    Returns:
        out (ndarray of bools) : de-spurred image.


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
def BOAarray(array:Union[xr.DataArray, xr.Dataset, np.ndarray, pd.DataFrame], dims: List[str] = ["time", "latitude", "longitude"]):
    """
    Converts the input array to a 3D xarray.DataArray with specified dimensions.
    If the input is already a 3D xarray.DataArray or Dataset, it will ensure that
    the dimensions are named according to the provided list.
    
    Args:
        array (Union[xr.DataArray, xr.Dataset, np.ndarray, pd.DataFrame]): The input array to be converted.
        dims (list): A list of dimension names to be applied to the resulting DataArray.
    Returns:
        xr.DataArray: A 3D xarray.DataArray with the specified dimensions.
    Raises:
        TypeError: If the input array is not of a supported type (xarray, numpy, or pandas).
    """
    # check dims
    if len(dims) != 3:
        raise ValueError("dims must be a list of three names: ['time','latitude','longitude']")
    # xarray
    if isinstance(array, (xr.core.dataarray.DataArray, xr.core.dataset.Dataset)):
        _arr = array
        n = 1
        while len(_arr.dims) < 3:
            _arr = _arr.expand_dims({f'tmp_{n}': 1})
            n += 1
        names_dict = {key: value for (key, value) in zip(_arr.dims, dims)}
        _arr = _arr.rename(names_dict)#[[dims]]
    # numpy
    elif isinstance(array, np.ndarray):
        _arr = array
        while _arr.ndim < 3:
            _arr = np.expand_dims(_arr, 0)
        _arr = xr.DataArray(_arr, dims=dims)
    # pandas
    elif isinstance(array, pd.core.frame.DataFrame):
        _arr = array
        _arr = xr.DataArray(_arr, dims=dims[1:]).expand_dims(dim=dims[0])
    #others
    else:
        raise TypeError(
            f"{type(_arr)} not supported. Switch to pandas, numpy or xarray arrays."
        )
    return _arr

# %% pyBOA class
@xr.register_dataarray_accessor("pyBOA")
@xr.register_dataset_accessor("pyBOA")
class pyBOA:
    """
    pyBOA class for processing and analyzing oceanographic data.
    This class provides methods for flagging peaks, applying morphological operations,
    calculating Sobel gradients, thresholding fronts, thinning data, rolling confidence intervals,
    rolling percentiles, and automatic detection of features in oceanographic datasets.
    
    Attributes:
        _array (xr.DataArray): The input data array to be processed.
    
    Use as follows:
    >>> array.pyBOA.auto_detection(rmse_target=0.01)
    """
    def __init__(self, array):
        """
        Initializes the pyBOA class with a given data array.
        
        Args:
            array (Union[xr.DataArray, xr.Dataset, np.ndarray, pd.DataFrame]): The input data array to be processed.
        Raises:
            TypeError: If the input array is not of a supported type (xarray, numpy, or pandas).
        """
        # dims
        self._dims = ["time", "latitude", "longitude"] # default
        self._array = BOAarray(array, self._dims)
        # buffer
        self._buffer_int = 1 #default
        self._buffer_ftprnt = (morphology.disk(self._buffer_int, dtype=np.float32)[np.newaxis,:]
                               if len(array.dims) == 3 else 
                               morphology.disk(self._buffer_int, dtype=np.float32)
                               )
        # dask
        self._use_dask = _HAS_DASK
        # verbose 
        self._verbose = True # default

    #%% Internal Helpers ############################################################
    def _init_logger(self):
        """Initialize a simple logger (called internally)."""
        if not hasattr(self, "_logger"):
            self._logger = logging.getLogger("pyBOA")
            handler = logging.StreamHandler()
            fmt = logging.Formatter("[%(name)s] %(levelname)s: %(message)s")
            handler.setFormatter(fmt)
            if not self._logger.handlers:
                self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)

    def _info(self, msg: str, level="info"):
        """Unified messaging for verbose mode."""
        if getattr(self, "_verbose", False):
            self._init_logger()
            prefix = "🐠" if level == "info" else "⚠️"
            log = getattr(self._logger, level, self._logger.info)
            log(f"{prefix} {msg}")


    def _buffer(self, array:xr.DataArray) -> np.ndarray:
        """
        Creates a buffer around NaN values in the input array using binary dilation.
        Args:
            array (xr.DataArray): The input data array.
        Returns:
            np.ndarray: A boolean array where True indicates a buffer around NaN values.
        """
        return morphology.binary_dilation(
            np.isnan(array), footprint=self._buffer_ftprnt)
    #%% set_* Helpers ############################################################
    def set_dims(self, dims:List[str] = ["time", "latitude", "longitude"]):
        if len(dims) != 3:
            raise ValueError("dims must be a list of three names: ['time','latitude','longitude']")
        self._dims = dims
        self._array = BOAarray(self._array, self._dims)
        return self

    def set_buffer(self, buffer:int = 1):
        self._buffer_int = buffer
        self._buffer_ftprnt = (morphology.disk(self._buffer_int, dtype=np.float32)[np.newaxis,:]
                               if len(self._array.dims) == 3 else 
                               morphology.disk(self._buffer_int, dtype=np.float32)
                               )
        return self

    def set_dask(self, verbose:bool = True):
        self._use_dask = bool(verbose)
        return self

    def set_verbose(self, verbose:bool = False):
        self._verbose = bool(verbose)
        return self
    
    def set_params(self, params: dict | None = None, **kwargs):
        # Merge params dict and kwargs
        params = params or {}
        params.update(kwargs)

        setter_map = {name.replace("set_", ""): method
                      for name, method in vars(self.__class__).items()
                      if callable(method) and name.startswith("set_") and (name != "set_params")
                      }
        
        for key, value in params.items():
            setter = setter_map.get(key)
            if setter:
                setter(self, value)  # note: pass self explicitly since we call unbound function
            else:
                raise KeyError(
                    f"Unknown parameter '{key}'. Available options: {list(setter_map)}"
                )
        return self

    #%% BOA calculations ############################################################
    def flag_n(self, n):
        """
        Flags peaks in the input array based on a rolling window of size n.
    
        Args:
            n (int): The size of the rolling window to be used for flagging peaks.
        Returns:
            xr.DataArray: boolean mask where values equal local window min or max.
        Raises:
            ValueError: If n is less than or equal to 0.
        """
        if n <= 0:
            raise ValueError("n must be positive integer")
        array = self._array.copy()
        window_size = {name: n for name in ["latitude", "longitude"]}
        window = array.rolling(window_size, center=True)
        peak_min = window.min(skipna=True)
        peak_max = window.max(skipna=True)
        flag = (peak_min == array) | (peak_max == array)
        return flag

    def mfNinM(self, m:int=5, n:int=3, return_filter:bool=False):
        """
        Applies a median filter to the input array using a large window size m and a small window size n.
        
        Args:
            m (int): The size of the large rolling window.
            n (int): The size of the small rolling window.
            return_filter (bool): If True, returns a tuple containing the filtered array and the filter mask.
        Returns:
            xr.DataArray or tuple: The filtered array, or a tuple containing the filtered array and the filter mask if return_filter is True.
        Raises:
            ValueError: If m is less than or equal to n.
        """
        # errors
        if m <= n:
            raise ValueError(
                f"Large window m:{m} can not be equal or smaller than small window n:{n}"
            )
        # preperation
        array = self._array.copy()
        self._info(f"mfNinM: applying median filter m={m}, n={n}")
        
        # mask
        peak_M = array.pyBOA.flag_n(m)
        peak_N = array.pyBOA.flag_n(n)
        to_filter = (peak_N & (~peak_M)).astype(bool) # or peak_N * ~peak_M

        # calculating medians      
        if _HAS_DASK and getattr(self, "_use_dask", False) and isinstance(array.data, da.Array):
            self._info("  Using Dask map_overlap for median filter")
            filtered = array.data.map_overlap(
                lambda block: median_filter(block, size=(1, n, n), mode="nearest"),
                depth=(0, n//2, n//2),
                boundary="reflect",
                dtype=float
            )
        else:
            self._info("  Using NumPy median_filter (in-memory)")
            filtered = xr.apply_ufunc(
                median_filter,
                array,
                kwargs={"size": (1, n, n), "mode": "nearest"},
                dask="parallelized" if _HAS_DASK and getattr(self, "_use_dask", False) else None,
                output_dtypes=[float],
            )

        # affecting medians where requiered
        filtered_da = xr.DataArray(filtered, coords=array.coords, dims=array.dims)
        result = xr.where(to_filter, filtered_da, array)

        return (result, to_filter) if return_filter else result

    def sobel_haversine(self):
        """
        Calculates the Sobel gradient of the input array using the haversine formula.
        The Sobel operator is applied to the array to compute the gradient in both horizontal and vertical directions,
        and the haversine formula is used to account for the curvature of the Earth.
        
        Returns:
            xr.DataArray: A DataArray containing the Sobel gradient of the input array.
        """
        # preparation
        array = self._array.copy()
        self._info("sobel_haversine: computing Sobel gradient (latitude-corrected)")
        
        # Latitude scaling factor
        lats = np.asarray(array.latitude)
        scale = np.cos(np.deg2rad(lats)).reshape((1, len(lats), 1))

        if _HAS_DASK and getattr(self, "_use_dask", False) and isinstance(array.data, da.Array):
            self._info("  Using Dask map_overlap for Sobel gradient")
            sobel_lat = array.data.map_overlap(sobel, depth={-2: 1}, boundary="reflect")
            sobel_lon = array.data.map_overlap(sobel, depth={-1: 1}, boundary="reflect")
            sobel_grad = scale * da.sqrt(sobel_lat**2 + sobel_lon**2)
        else:
            self._info("  Using NumPy Sobel (in-memory)")
            vals = array.values.astype(float)
            if vals.ndim == 2:
                vals = vals[np.newaxis, ...]
            sobel_lat = sobel(vals, axis=1, mode="nearest")
            sobel_lon = sobel(vals, axis=2, mode="nearest")
            sobel_grad = scale * np.hypot(sobel_lat, sobel_lon)
            if self._array.values.ndim == 2:
                sobel_grad = np.squeeze(sobel_grad, axis=0)

        # Check dims
        if self._array.values.ndim == 2:
            sobel_grad = np.squeeze(sobel_grad, axis=0)

        sobel_da = xr.DataArray(sobel_grad, coords=array.coords, dims=array.dims)
        return sobel_da

    def front_trsh(self, wndw:int=64, prcnt:int=90):
        """
        Thresholds the fronts in the input array based on a rolling window and a specified percentile.
        
        Args:
            wndw (int): The size of the rolling window to be used for thresholding.
            prcnt (int): The percentile value to be used for thresholding.
        Returns:
            xr.DataArray: A DataArray containing the thresholded fronts.
        """
        # preparation
        array = self._array.copy()
        self._info(f"front_trsh: wndw={wndw}, prcnt={prcnt}")
        use_dask = _HAS_DASK and getattr(self, "_use_dask", False) and isinstance(array.data, da.Array)
        def _percentile_func(x, q):
            if use_dask:
                return da.nanpercentile(x, q=q, axis=(-2, -1))
            else:
                return np.nanpercentile(x, q=q, axis=(-2, -1))

        # rolling array        
        window = array.rolling(
            {name: wndw for name in ["latitude", "longitude"]}, center=True, min_periods=1
            ).construct(latitude="lat_win", longitude="lon_win")
        
        self._info("  Computing rolling percentile (Dask lazy)" if use_dask else "  Computing rolling percentile (NumPy)")

        window_qt = xr.apply_ufunc(
            _percentile_func,
            window,
            kwargs={"q": prcnt},
            input_core_dims=[["lat_win", "lon_win"]],
            output_core_dims=[[]],
            vectorize=False,
            dask="parallelized" if use_dask else None,
            output_dtypes=[float],
            )
        frnt = (array > window_qt).astype(float)
        return frnt

    def thinning(self, iteration:int=2, min_size:int=7, f_dilate:bool=False):
        """
        Applies morphological thinning to the input array.

        Args:
            iteration (int): The number of iterations for morphological thinning.
            min_size (int): The minimum size of objects to be retained after thinning.
            f_dilate (bool): If True, applies dilation before thinning.
        Returns:
            xr.DataArray: A DataArray containing the thinned fronts.
        """
        array = self._array.copy()
        self._info(f"thinning: processing {len(array.coords.get('time', [None]))} time steps (iteration={iteration})")
        
        def _processing(block):
            # ensure 3d
            if block.ndim == 2:
                block = block[np.newaxis, ...]
            out_block = np.empty_like(block, dtype=block.dtype)
            for t in range(block.shape[0]):
                frnt = block[t].astype(bool).copy()
                for it in range(iteration):
                    frnt = morphology.thin(frnt) # morphological thining
                    frnt = spur(frnt.astype(np.uint8), n_iter=1) # spur removal expect binary 0/1
                    frnt = morphology.remove_small_objects(frnt.astype(bool), min_size=min_size, connectivity=2) # clean small object
                    frnt = morphology.remove_small_holes(frnt)# remove small holes
                    if it < iteration - 1:
                        frnt = morphology.dilation(frnt) # dilate
            frnt = morphology.thin(frnt)
            out_block[t] = frnt.astype(block.dtype)
            # Restore dims: if original 2D, squeeze time
            if self._array.values.ndim == 2:
                out_block = np.squeeze(out_block, axis=0)
            return out_block

        #isolate values for speed
        if _HAS_DASK and getattr(self, "_use_dask", False) and isinstance(array.data, da.Array):
            self._info("  Applying thinning lazily using Dask map_blocks")
            out = da.map_blocks(_processing, array.data, dtype=array.dtype)
        else:
            self._info("  Applying thinning eagerly using NumPy (memory heavy)")
            vals = np.asarray(array.data)
            out = _processing(vals)
        
        out_da = xr.DataArray(out, coords=array.coords, dims=array.dims)
        return out_da

    #%% pyBOA auto_detection ############################################################ 
    def auto_detection(self, rmse_target: float = 0.01,
                       m:int=5, n:int=3, #return_filter:bool=False, # mfNinM parameters
                       wndw:int=64, prcnt:int=90, # front_trsh parameters
                       iteration:int=2, min_size:int=7, f_dilate:bool=False # thinning parameters
                       ):
        """
        Automatically detects features in the input array by applying a median filter,
        calculating Sobel gradients, thresholding fronts, and thinning the data.
        
        Args:
            rmse_target (float): The target root mean square error for the median filter.
            m (int): The size of the large rolling window for the median filter.
            n (int): The size of the small rolling window for the median filter.
            return_filter (bool): If True, returns the filtered array and the filter mask.
            wndw (int): The size of the rolling window for thresholding fronts.
            prcnt (int): The percentile value for thresholding fronts.
            iteration (int): The number of iterations for morphological thinning.
            min_size (int): The minimum size of objects to be retained after thinning.
            f_dilate (bool): If True, applies dilation before thinning.
        Returns:
            xr.DataArray: A DataArray containing the filtered, Sobel, and front-detected data.
        Raises:
            ValueError: If rmse_target is less than or equal to 0.
        """
        # error
        if rmse_target <= 0:
            raise ValueError("rmse_target must be positive")
        
        # preparations
        if isinstance(self._array, xr.core.dataarray.DataArray):
            ds_in = self._array.to_dataset(name=self._array.name) if (hasattr(self._array, "name") and self._array.name) else self._array.to_dataset()
        else:
            ds_in = self._array
        dataset_copy = self._array.copy(deep=True)
        vars_to_process = list(ds_in.data_vars)
        self._info(f"auto_detection: processing variables: {vars_to_process}")

        # vars loop 
        for v in ds_in.data_vars:
            self._info(f"auto_detection: variable {v}, rmse target={rmse_target}")
            arr = ds_in[v]
            arr_work = arr.copy()
            rmse = np.inf

            # mfNinM until convergence
            while rmse > rmse_target:
                self._info(f"  Filtering iteration: current rmse={rmse:.6g}")
                res_fltrd = arr_work.pyBOA.mfNinM(m, n , return_filter = False)
                # differences projected vs measures
                delta = np.subtract(res_fltrd, arr_work)
                rmse_num, rmse_denom = np.nansum(delta ** 2), np.nansum(np.where(np.isfinite(delta), 1.0, 0.0))
                rmse = 0.0 if rmse_denom==0 else np.sqrt(rmse_num / rmse_denom)   

                arr_work = res_fltrd.copy()
                # safety: break if no change
                if np.allclose(res_fltrd.values, arr_work.values, equal_nan=True):
                    break

            # sobel
            mask_buffer = ~self._buffer(arr)
            res_sobel = res_fltrd.where(mask_buffer).pyBOA.sobel_haversine()
            res_blob = res_sobel.pyBOA.front_trsh(wndw, prcnt)
            res_frnt = res_blob.pyBOA.thinning(iteration, min_size, f_dilate)

            # strore result in dataset copy
            dataset_copy[f"{v}_filtered"] = (arr.dims, res_fltrd.data)
            dataset_copy[f"{v}_sobel"] = (arr.dims, res_sobel.data)
            dataset_copy[f"{v}_fronts"] = (arr.dims, res_frnt.where(res_frnt > 0).data)

            self._info(f"auto_detection: finished, rmse final={rmse:.6g}")
        return dataset_copy
