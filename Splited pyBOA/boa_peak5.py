from numpy import nanmax, nanmin, isnan, zeros_like, shape, argwhere, array_equal, asarray
from xarray import DataArray

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
