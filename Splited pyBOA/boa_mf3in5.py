from numpy import nanmedian, where as npwhere
from boa_flag_n import flag_n
#%% mf3in5
def mf3in5(data_nc, peak5, peak3):
    '''
    Contextual median filtering
    Input:
        data_nc:     .nc dataset/dataArray
    Output:
        filtered_nc: median filtered data_nc
    WARNING: As per BOA design, works with 2 implementation of flag_n with n = 5 and n = 5
    '''
    peak_5 = flag_n(data_nc, 5)
    peak_3 = flag_n(data_nc, 3)
    to_filter = peak_3 * ~peak_5
    filtered_nc = data_nc.copy()
    idx = npwhere(to_filter)

    for it, ix, iy in zip(*idx):
        window = data_nc[it, ix-1:ix+2, iy-1:iy+2]
        filtered_nc[it, ix, iy] = nanmedian(window)

    return filtered_nc
