from numpy import cos, sqrt, meshgrid
from xarray import DataArray
from math import pi
from scipy.ndimage import sobel

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
