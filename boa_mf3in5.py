# -*- coding: utf-8 -*-
"""
Created on Wed May  4 08:37:25 2022

@author: alhe551
"""
from numpy import nanmedian, copy

def mf3in5(data_nc, peak5, peak3):
    '''
    Contextual median filtering
    Input:
        data_nc:     .nc file
        peak5:       output from boa_peak5
        peak3:       output from boa_peak3
    Output:
        filtered_nc: median filtered data_nc
    '''
    nc_shape = data_nc.shape
    range_lat = range(2,nc_shape[0]-2,1)
    range_lon = range(2,nc_shape[1]-2,1)
    filtered_nc = copy(data_nc)

    for i in range_lat:
        for j in range_lon:
            # window
            window = filtered_nc[i-1:i+2,j-1:j+2]
            
            if peak3[i,j] == 1 and peak5[i,j] == 0:
                filtered_nc[i,j] = nanmedian(window)

    return filtered_nc

