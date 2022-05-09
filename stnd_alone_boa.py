# -*- coding: utf-8 -*-
"""
Created on Wed May  4 12:35:19 2022

@author: alhe551
"""

#Environment setting
import os
from warnings import simplefilter

import numpy as np
import xarray as xr
from math import sqrt
from scipy.ndimage import sobel
from tqdm import tqdm

print('Importing packages ...')
PATH_ROOT = os.path.dirname(os.path.realpath(__file__))
os.chdir(PATH_ROOT)

#keep to avoid runtimewarnings when all NAN slices
simplefilter("ignore", category=RuntimeWarning)

#%% Variables and functions
print('\nLoading custom functions ...')

from boa_mf3in5 import mf3in5
from boa_peak3 import peak_3
from boa_peak5 import peak_5
#%% Extraction nc
print('\nExtracting nc file ...')

nc_file = 'CHL_L3_REP_OBSERVATIONS_4km_2022_04_27.nc'
nc = xr.open_dataset('CHL_L3_REP_OBSERVATIONS_4km_2022_04_27.nc').load()
# mapping "Equirectangular"

# REMOVE LATER
#nc = nc.sel(time = slice('2012-01-01', '2012-01-03'))

# Isolating the log(CHL)
nc = np.log(nc)

#%% detection
print('\nBeginning detection ...')

# RMSE definition
rmse_target = 0.01
print(f'Current target of the RMSE: {rmse_target}.')

# Detection loop
for day in tqdm(nc.time.values):
    rmse = 1
    CHLa = nc.CHL.sel(time=day)
    while rmse > rmse_target:

        res_peak5 = peak_5(CHLa)  # 5*5 window
        res_peak3 = peak_3(CHLa) # 3*3 window
        res_fltrd = mf3in5(CHLa, res_peak5, res_peak3) # contextual filter

        delta_nc = np.subtract(res_fltrd ,CHLa) # differences projected vs measures
        rmse = sqrt(np.nansum(delta_nc**2)/np.nansum(delta_nc/delta_nc)) # RMSE
        
        CHLa = np.copy(res_fltrd) # Reafecting 
        #rmse = 0 # Security for one run only

    print(f' RMSE = {rmse}')

    # Sobel kernels
    sobel_vrt = sobel(res_fltrd,0) # Sobel along the latitude?
    sobel_hzt = sobel(res_fltrd,1) # Sobel along the longitude?
    sobel_grd = np.sqrt(sobel_hzt**2 + sobel_vrt**2)

    # Matching DataArray
    res_fltrd = xr.DataArray(res_fltrd, dims = ['lat', 'lon']).expand_dims({'time':1})
    res_peak5 = xr.DataArray(res_peak5, dims = ['lat', 'lon']).expand_dims({'time':1})
    res_peak3 = xr.DataArray(res_peak3, dims = ['lat', 'lon']).expand_dims({'time':1})
    sobel_grd = xr.DataArray(sobel_grd, dims = ['lat', 'lon']).expand_dims({'time':1})

    # Saving
    if day == nc.time.values.min():
        nc_fltrd = res_fltrd
        nc_peak5 = res_peak5
        nc_peak3 = res_peak3
        nc_sobel = sobel_grd
    else:
        nc_fltrd = xr.concat([nc_fltrd, res_fltrd], dim='time')
        nc_peak5 = xr.concat([nc_peak5, res_peak5], dim='time')
        nc_peak3 = xr.concat([nc_peak3, res_peak3], dim='time')
        nc_sobel = xr.concat([nc_sobel, sobel_grd], dim='time')

# Coupling to original nc for save and display purposes
#nc['fltrd'] = (['time', 'lat', 'lon'], nc_fltrd.data)
nc['peak5'] = (['time', 'lat', 'lon'], nc_peak5.data)
nc['peak3'] = (['time', 'lat', 'lon'], nc_peak3.data)
nc['sobel'] = (['time', 'lat', 'lon'], nc_sobel.data)

nc.to_netcdf('CHL_L3_REP_4km_2022_04_27_DETECTION.nc','w')

#%% End
print('End.','\a')