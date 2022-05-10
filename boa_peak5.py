from numpy import nanmax, nanmin, zeros_like, shape, argwhere, array_equal, isnan

def peak_5(data_nc):
    '''
    Detection of extrema in 5*5 windows
    Input:
        data_nc:   .nc dataset
    Output:
        peak5:     data sheet with data_nc dimensions
    Issues may occure if dimmensions order is not [lat, lon, other] or [lon, lat, other]. Hence, 2D arrays are prefered, like [lat, lon].
    '''
    nc_shape = shape(data_nc)
    range_lat = range(2,nc_shape[0]-2,1)
    range_lon = range(2,nc_shape[1]-2,1)
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

    return peak5
