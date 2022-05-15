# -*- coding: utf-8 -*-
"""
Created on Fri May 13 11:31:52 2022

@author: alhe551
"""

# Packages



from numpy import nanmax, nanmin, isnan, nanmedian, cos, sqrt
from numpy import zeros_like, shape, argwhere, array_equal, asarray, array, meshgrid

from xarray import DataArray
from math import pi
from scipy.ndimage import sobel


#%% peak_5
def peak_5(data_nc):
    '''
    Detection of extrema in 5*5 windows
    Input:
        data_nc:   .nc dataset
    Output:
        peak5:     data sheet with data_nc dimensions
    
    WARNING: This function was built for CMEMS dataset with dims [time, lat, lon]. 
             You may edit the idexes to match your own dims order.
    '''
    dimensions = data_nc.dims
    coordinates = data_nc.coords
    data_nc = asarray(data_nc)
    nc_shape = shape(data_nc)
    range_lat = range(2,nc_shape[-2]-2,1)
    range_lon = range(2,nc_shape[-1]-2,1)
    peak5 = zeros_like(data_nc)

    for i in range_lat:
        for j in range_lon:
            # window
            window = data_nc[i-2:i+3,j-2:j+3]

            # nan filtering
            if isnan(window.all()):
                pass

            # extracting min and max position in the array
            else:
                peak_min = array_equal([[2,2]], argwhere(window == nanmin(window)))
                peak_max = array_equal([[2,2]], argwhere(window == nanmax(window)))

                if peak_min | peak_max:
                    peak5[i,j] = 1
    peak5 = DataArray(peak5, coords = coordinates, dims = dimensions)

    return peak5

#%% peak_3
def peak_3(data_nc):
    '''
    Detection of extrema in 3*3 windows
    Input:
        data_nc:   .nc dataset
    Output:
        peak3:     data sheet with data_nc dimensions
   
    WARNING: This function was built for CMEMS dataset with dims [time, lat, lon]. 
             You may edit the idexes to match your own dims order.
    '''
    dimensions = data_nc.dims
    coordinates = data_nc.coords
    data_nc = asarray(data_nc)
    nc_shape = shape(data_nc)
    range_lat = range(2,nc_shape[-2]-2,1)
    range_lon = range(2,nc_shape[-1]-2,1)
    peak3 = zeros_like(data_nc)
    
    for i in range_lat:
        for j in range_lon:
            # window
            window = data_nc[i-2:i+3,j-2:j+3]

            # nan filtering
            if isnan(window.all()):
                pass

            # extracting min and max position in the array
            else:
                peak_min = array_equal([[1,1]], argwhere(window == nanmin(window)))
                peak_max = array_equal([[1,1]], argwhere(window == nanmax(window)))

                if peak_min | peak_max:
                    peak3[i,j] = 1
    peak3 = DataArray(peak3, coords = coordinates, dims = dimensions)

    return peak3

#%% mf3in5
def mf3in5(data_nc, peak5, peak3):
    '''
    Contextual median filtering
    Input:
        data_nc:     .nc file
        peak5:       output from boa_peak5
        peak3:       output from boa_peak3
    Output:
        filtered_nc: median filtered data_nc
    WARNING: The input order of peak5 and peak3 matters. Reversing it leads to wrong results
             This function was built for CMEMS dataset with dims [time, lat, lon]. 
             You may edit the idexes to match your own dims order.
    '''
    nc_shape = data_nc.shape
    range_lat = range(2,nc_shape[-2]-2,1)
    range_lon = range(2,nc_shape[-1]-2,1)
    filtered_nc = data_nc.copy()

    for i in range_lat:
        for j in range_lon:
            # window
            window = filtered_nc[i-1:i+2,j-1:j+2]
            
            if peak3[i,j] == 1 and peak5[i,j] == 0:
                filtered_nc[i,j] = nanmedian(window)
    
    filtered_nc = DataArray(filtered_nc, dims = data_nc.dims)
    
    return filtered_nc

#%% sobel_haversine
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

    sobel_hzt = sobel(data_nc,0) # Sobel along the latitude
    sobel_vrt = hvrsn*sobel(data_nc,1) # Sobel along the longitude

    sobel_grd = sqrt(sobel_hzt**2 + sobel_vrt**2) #gradient calculation
    sobel_grd = DataArray(sobel_grd, coords = coordinates, dims = dimensions)
    return sobel_grd
