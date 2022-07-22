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