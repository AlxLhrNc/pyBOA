# -*- coding: utf-8 -*-
"""
Created on Wed May  4 12:35:19 2022

@author: alhe551

To execute this file, the files boa_peak3.py, boa_peak5.py and boa_m3in5.py are necessary. A sample file is available.
"""

#Environment setting
import os
from warnings import simplefilter

import numpy as np
import xarray as xr
from math import sqrt
from scipy.ndimage import sobel
from tqdm import tqdm
from time import sleep

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
from boa_sobel_haversine import sobel_haversine

#or

from pyBOA import peak_5, peak_3, mf3in5, sobel_haversine

#%% Extraction nc
print('\nExtracting nc file ...')

nc_file = 'CHL_4km_sample.nc' 
nc = xr.open_dataset(nc_file).load()

# Isolating the log(CHL)
nc = np.log(nc)

#%% Detection
print('\nBeginning detection ...')

# RMSE definition
rmse_target = 0.01
print(f'Current target of the RMSE: {rmse_target}.')

# Detection loop
for day in tqdm(nc.time.values):
    rmse = 1
    CHLa = nc.sel(time=day)
    print('') #aesthetic with tqdm if intermediary RMSE
    while rmse > rmse_target:

        res_peak5 = peak_5(CHLa)  # 5*5 window
        res_peak3 = peak_3(CHLa) # 3*3 window
        res_fltrd = mf3in5(CHLa, res_peak5, res_peak3) # contextual filter

        delta_nc = np.subtract(res_fltrd ,CHLa) # differences projected vs measures
        rmse = sqrt(np.nansum(delta_nc**2)/np.nansum(delta_nc/delta_nc)) # RMSE
        print(f' RMSE = {rmse}') # print intermediary RMSE, uncomment to see
        CHLa = np.copy(res_fltrd) # Reafecting 
        #rmse = 0 # Security for one run only
    sleep(.2) #aesthetic with tqdm if intermediary RMSE
    #print(f'\n RMSE = {rmse}') # print final RMSE, uncomment to see

    # Custom sobel
    res_sobel = sobel_haversine2(res_fltrd)

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

save_file = nc_file.replace('.nc','_DETECTION.nc')
nc.to_netcdf(save_file,'w')

#%% End
print('End.','\a')
