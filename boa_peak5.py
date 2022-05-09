from numpy import nanmax, nanmin, zeros_like, shape

def peak_5(data_nc):
    '''
    Detection of extrema in 5*5 windows
    Input:
        data_nc:   .nc file
    Output:
        peak5:     data sheet with data_nc dimensions
    '''
    nc_shape = data_nc.shape
    range_lat = range(2,nc_shape[0]-2,1)
    range_lon = range(2,nc_shape[1]-2,1)
    peak5 = zeros_like(data_nc)

    for i in range_lat:
        for j in range_lon:
            # window
            window = data_nc[i-2:i+3,j-2:j+3]

            #center
            center = window[2,2]

            # W-E
            slice_w_e = window[2,:]
            # N-S
            slice_n_s = window[:,2]
            # NW-SE
            slice_nw_se = [window[0,4],window[1,3],window[2,2],window[3,1],window[4,0]]
            # NE-SW
            slice_ne_sw = [window[0,0],window[1,1],window[2,2],window[3,3],window[4,4]]

            all_min = [nanmin(slice_w_e), nanmin(slice_n_s),
                       nanmin(slice_nw_se), nanmin(slice_ne_sw)]
            all_max = [nanmax(slice_w_e), nanmax(slice_n_s),
                       nanmax(slice_nw_se), nanmax(slice_ne_sw)]
            
            peak_min = all(peak == center for peak in all_min)
            peak_max = all(peak == center for peak in all_max)

            if peak_min | peak_max:
                peak5[i,j] = 1


    return peak5
