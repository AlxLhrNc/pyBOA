from numpy import nanmax, nanmin, zeros_like, shape, argwhere, array_equal, isnan

def peak_3(data_nc):
    '''
    Detection of extrema in 3*3 windows
    Input:
        data_nc:   .nc dataset
    Output:
        peak3:     data sheet with data_nc dimensions
    Issues may occure if dimmensions order is not [lat, lon, other] or [lon, lat, other]. Hence, 2D arrays are prefered, like [lat, lon].
    '''
    nc_shape = shape(data_nc)
    range_lat = range(2,nc_shape[0]-2,1)
    range_lon = range(2,nc_shape[1]-2,1)
    peak3 = zeros_like(data_nc)
    
    for i in range_lat:
        for j in range_lon:
            # window
            window = data_nc[i-1:i+2,j-1:j+2]

            # nan filtering
            if isnan(window.all()):
                pass

            # extracting min and max position in the array
            else:
                peak_min = array_equal([[1,1]], argwhere(window == nanmin(window)))
                peak_max = array_equal([[1,1]], argwhere(window == nanmax(window)))

                if peak_min | peak_max:
                    peak3[i,j] = 1

    return peak3
