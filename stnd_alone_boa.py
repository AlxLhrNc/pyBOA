# -*- coding: utf-8 -*-
"""
Created on Wed May  4 12:35:19 2022

@author: AlxndrLhr

To execute this file, the files boa_*.py or pyBOA.py are necessary. A sample file is available.
"""

#Environment setting
import os
from warnings import simplefilter

import numpy as np
import xarray as xr
from math import sqrt

print('Importing packages ...')
PATH_ROOT = os.path.dirname(os.path.realpath(__file__))
os.chdir(PATH_ROOT)

#keep to avoid runtimewarnings when all NAN slices
simplefilter("ignore", category=RuntimeWarning)

#%% Variables and functions
print('\nLoading custom functions ...')

#from boa_mf3in5 import mf3in5
#from flag_n import flag_n
#from boa_sobel_haversine import sobel_haversine

#or

from pyBOA import flag_n, mf3in5, sobel_haversine

#%% Extraction nc
print('\nExtracting nc file ...')

nc_file = 'CHL_4km_sample.nc' 
nc = xr.open_dataarray(nc_file).load()

# Isolating the log(CHL)
#nc = np.log(nc)

#%% Detection
print('\nBeginning detection ...')

# RMSE definition
rmse_target = 0.01
print(f'Current target of the RMSE: {rmse_target}.')

# Detection loop

rmse = 1
CHLa = nc.copy()
print('') #aesthetic with tqdm if intermediary RMSE
while rmse > rmse_target:
    res_peak5 = flag_n(CHLa, 5)  # 5*5 window if you need to save the data
    res_peak3 = flag_n(CHLa, 3) # 3*3 window if you need to save the data
    res_fltrd = mf3in5(CHLa) # contextual filter

    delta_nc = np.subtract(res_fltrd ,CHLa) # differences projected vs measures
    rmse = sqrt(np.nansum(delta_nc**2)/np.nansum(delta_nc/delta_nc)) # RMSE
    print(f' RMSE = {rmse}') # print intermediary RMSE, uncomment to see
    CHLa = res_fltrd.copy() # Reafecting 
    #rmse = 0 # Security for one run only
#print(f'\n RMSE = {rmse}') # print final RMSE, uncomment to see

# Custom sobel
res_sobel = sobel_haversine(res_fltrd)

# Coupling to original nc for save and display purposes
#nc['fltrd'] = (['time', 'lat', 'lon'], nc_fltrd.data)
nc['peak5'] = (['time', 'lat', 'lon'], res_peak5.data)
nc['peak3'] = (['time', 'lat', 'lon'], res_peak3.data)
nc['sobel'] = (['time', 'lat', 'lon'], res_sobel.data)

save_file = nc_file.replace('.nc','_DETECTION.nc')
nc.to_netcdf(save_file,'w')

#%% End
print('End.','\a')
