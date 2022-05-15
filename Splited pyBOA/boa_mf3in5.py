from numpy import nanmedian
from xarray import DataArray

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
