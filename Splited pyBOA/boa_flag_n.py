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
