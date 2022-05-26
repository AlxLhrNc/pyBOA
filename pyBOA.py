# -*- coding: utf-8 -*-
"""
Created on Fri May 13 11:31:52 2022

@author: alhe551
"""

# Packages

from numpy import isnan, nanmedian, cos, sqrt, nanmean, nansum, nan as npnan
from numpy import meshgrid, histogram, ravel
from numpy import where as npwhere
from xarray import DataArray, where as xrwhere
from math import pi
from scipy.ndimage import sobel

#%% flag_n
def flag_n(data_nc, n):
    '''
    Detection of extrema in n*n windows
    Input:
        data_nc:   .nc dataset/dataArray
    Output:
        peak5:     data sheet with data_nc dimensions
    
    WARNING: This function was built for CMEMS dataset with dims [time, lat, lon]. 
             You may edit the names to match your own dims.
    '''
    
    window_size = {name: n for name in ['lat', 'lon']}
    window = data_nc.rolling(window_size, center=True)

    peak_min = window.min()
    peak_max = window.max()

    flag = (peak_min == data_nc) | (peak_max == data_nc)

    return flag

#%% mf3in5
def mf3in5(data_nc, peak5, peak3):
    '''
    Contextual median filtering
    Input:
        data_nc:     .nc dataset/dataArray
    Output:
        filtered_nc: median filtered data_nc
    WARNING: As per BOA design, works with 2 implementation of flag_n with n = 5 and n = 5
    '''
    peak_5 = flag_n(data_nc, 5)
    peak_3 = flag_n(data_nc, 3)
    to_filter = peak_3 * ~peak_5
    filtered_nc = data_nc.copy()
    idx = npwhere(to_filter)

    for it, ix, iy in zip(*idx):
        window = data_nc[it, ix-1:ix+2, iy-1:iy+2]
        filtered_nc[it, ix, iy] = nanmedian(window)

    return filtered_nc

#%% sobel_haversine (faster)
def sobel_haversine(data_nc):
    '''
    Sobel filter adjusted using Haversine formula to deal with lon/lat to deal with distances change
    Input:
        data_nc:  .nc dataset/dataArray
    Output:
        output:   sobel firltered dataArray with data_nc dimensions
    '''
    dimensions = data_nc.dims
    coordinates = data_nc.coords

    tmp, hvrsn = meshgrid(data_nc.lon,cos(data_nc.lat*pi/180)) # extracting cos(lat) as a matrix

    sobel_hzt, sobel_vrt = sobel(data_nc,1),sobel(data_nc,2) # Sobel along the longitude

    sobel_grd = hvrsn*sqrt(sobel_hzt**2 + sobel_vrt**2) #gradient calculation
    sobel_grd = DataArray(sobel_grd, coords = coordinates, dims = dimensions)
    return sobel_grd

#%% otsu
def otsu_thrshld(data_nc):
    '''
    Otsu automated thresholding (DOI: 10.1109/TSMC.1979.4310076)
    Input:
        data_nc:  .nc dataset/dataArray
    Output:
        thrshld_f:  final threshold
        otsu_out:   binary dataArray with data_nc dimensions
    '''
    N = data_nc.shape[-2]*data_nc.shape[-1]
    weight = 1.0/N
    rvl = ravel(data_nc)[~isnan(ravel(data_nc))]
    hist, bins = histogram(rvl, 256)
    #loking to maximize nu with nu = var_between_class/var_total. OR minimize var_between_class
    thrshld_f = -1
    value_f = -1
    for k in bins:
        ind = int(npwhere(bins==k)[0])
        Wb = nansum(hist[:ind]) * weight
        Wf = nansum(hist[ind:]) * weight
        
        muB = nanmean(hist[:ind]) * weight
        muF = nanmean(hist[ind:]) * weight

        value = Wb * Wf * (muB - muF) ** 2
        
        if value > value_f:
                thrshld_f = k
                value_f = value
    otsu_out = data_nc.copy()
    otsu_out = xrwhere(otsu_out>thrshld_f,1,npnan)
    otsu_out = xrwhere(isnan(data_nc),npnan,otsu_out)    

    return round(thrshld_f,2), otsu_out
