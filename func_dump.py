import numpy as np
from scipy.special import kv,gamma

def get_pairwise_geo_distance(lon,lat):
    # Calculate spherical distance
    N = len(lon)
    pd = np.zeros((N,N))
    coslat = np.cos(lat/180.0*np.pi)
    sinlat = np.sin(lat/180.0*np.pi)
    
    for i in range(N):
        dx = lon - lon[i]
        cosdx = np.cos(dx/180.0*np.pi)
        pd[i,:] = coslat[i]*coslat*cosdx + sinlat[i]*sinlat
    pd[pd>1] = 1
    pd = 180.0/np.pi*np.arccos(pd)
    np.fill_diagonal(pd,0.0)
    return pd

def get_pairwise_cross_distance(lon1,lat1,lon2,lat2):
    N1 = len(lon1)
    N2 = len(lon2)
    
    coslat1 = np.cos(lat1/180.0*np.pi)
    sinlat1 = np.sin(lat1/180.0*np.pi)

    coslat2 = np.cos(lat2/180.0*np.pi)
    sinlat2 = np.sin(lat2/180.0*np.pi)
    
    pd = np.zeros((N1,N2))
    for i in range(N1):
        dx = lon2 - lon1[i]
        cosdx = np.cos(dx/180.0*np.pi)
        pd[i,:] = coslat1[i] * coslat2 * cosdx + sinlat1[i] * sinlat2
    pd[pd>1] = 1
    pd = 180.0/np.pi*np.arccos(pd)
    return pd
    
def get_all_geo_distance(lon,lat,lon0,lat0):
    N = len(lon)
    coslat = np.cos(lat/180.0*np.pi)
    sinlat = np.sin(lat/180.0*np.pi)
    
    coslat0 = np.cos(lat0/180.0*np.pi)
    sinlat0 = np.sin(lat0/180.0*np.pi)
        

    dx = lon - lon0
    cosdx = np.cos(dx/180.0*np.pi)
    pd = coslat0*coslat*cosdx + sinlat0*sinlat
    pd[pd>1] = 1
    pd = 180.0/np.pi*np.arccos(pd)
    return pd

def C_sph_nugget(d,covarParams):
    """Evaluate spherical covariance function
    """
    C = np.zeros(d.shape)
    if covarParams[2] == 0.0:
        np.fill_diagonal(C,covarParams[0] + covarParams[1])
    else:
        C[d==0] = covarParams[0] + covarParams[1]
        C[(d>0)&(d<covarParams[2])] =  covarParams[1] * (1 - 1.5*d[(d>0)&(d<covarParams[2])]/covarParams[2] + 0.5 * d[(d>0)&(d<covarParams[2])]**3  / covarParams[2]**3)
        C[(d>0)&(d>=covarParams[2])] =  0.0
    return C

def get_semivariogram(pd,val,h,binWidth):
    N = pd.shape[0]
    valid = np.where((pd >= h-0.5*binWidth) & (pd <= h+0.5*binWidth))
    dz = (val[valid[0]] - val[valid[1]])**2
    return np.sum(dz) / (2.0 * len(dz)),len(dz)/2

def get_semivariogram_lags(pd,val,hmax,binWidth):
    nbins = int(hmax/binWidth) + 1
    hs = np.arange(0,nbins,1) * binWidth + 0.5 * binWidth
    gamma = np.zeros((nbins))
    bincount = np.zeros((nbins))
    for i in range(nbins):
        gamma[i],bincount[i] = get_semivariogram(pd,val,hs[i],binWidth)
    return hs,gamma,bincount