from numpy import nanmax, nanmin, zeros_like

def peak_3(data_nc):
    '''
    Detection of extrema in 3*3 windows
    Input:
        data_nc:   .nc dataset
    Output:
        peak3:     data sheet with data_nc dimensions
    '''
    nc_shape = data_nc.shape
    range_lat = range(2,nc_shape[0]-2,1)
    range_lon = range(2,nc_shape[1]-2,1)
    peak3 = zeros_like(data_nc)
    
    for i in range_lat:
        for j in range_lon:
            # window
            window = data_nc[i-1:i+2,j-1:j+2]

            #center
            center = window[1,1]

            peak_min = center == nanmin(window)
            peak_max = center == nanmax(window)

            if peak_min | peak_max:
                peak3[i,j] = 1

    return peak3
