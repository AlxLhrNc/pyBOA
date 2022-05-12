from numpy import shape, array, zeros_like, cos, sqrt
from math import radians
from xarray import DataArray
def sobel_haversine(data_nc, lat_key = 'lat'):
    '''
    Sobel filter adjusted using Haversine formula to deal with lon/lat to deal with distances change
    
		Input:
        data_nc:  .nc dataset/dataArray
        lat_key:  key for latitude in data_nc (default CMEMS 'lat')
    Output:
        output:   sobel firltered dataArray with data_nc dimensions
    
		WARNING: This function was built for CMEMS dataset with dims [time, lat, lon]. 
             You may edit the idexes to match your own dims order.
    '''
    nc_shape = shape(data_nc)
    range_lat = range(1,nc_shape[1]-1,1)
    range_lon = range(1,nc_shape[2]-1,1)
    output = zeros_like(data_nc)
    
    Gx = array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
    Gy = array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
    for i in range_lat:
        for j in range_lon:
            hvrsn = cos(radians(data_nc[lat_key][i])).values # haversine cos(lat) part
            gx = ((Gx*hvrsn)*data_nc[0,i-1:i+2, j-1:j+2]).sum()  # x direction
            gy = (Gy*data_nc[0,i-1:i+2, j-1:j+2]).sum()  # y direction
            output[0,i,j] = sqrt(gx**2 + gy**2)  # gradient
    output = DataArray(output, dims = data_nc.dims)
    return output
