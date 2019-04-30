import numpy as np
import matplotlib.pyplot as plt
import clean_kriging as clean_kriging
from func_dump import get_pairwise_geo_distance
import sklearn.cluster as cluster
import datetime


def test_cluster_size(point_data,max_size,do_plot=False,chosen_range=None,
        perc_levels=20):
    """Test effect of number of clusters on cluster radius and size
    """
    
    cluster_sizes = range(5,max_size,1)
    radius_1 = np.zeros((len(cluster_sizes),3))
    cluster_N = np.zeros((len(cluster_sizes),3))
    percentages = np.zeros((len(cluster_sizes),perc_levels+1))

    X = point_data
    Xsel = X
    pd = get_pairwise_geo_distance(Xsel[:,0],Xsel[:,1]) 

    for k,n_clusters in enumerate(cluster_sizes):
        model = cluster.AgglomerativeClustering(linkage='complete',
                                                affinity='precomputed',
                                                n_clusters=n_clusters)
        model.fit(pd)
        radius = np.zeros((n_clusters))
        cluster_members = np.zeros((n_clusters))
        for i,c in enumerate(np.unique(model.labels_)):
            ix = np.where(model.labels_==c)[0]
            radius[i] = 0.5*pd[np.ix_(ix,ix)].max()
            cluster_members[i] = np.sum(model.labels_==c)
        r1i,r1a,r1s = (radius.min(),radius.max(),radius.std())
        radius_1[k,0] = r1i
        radius_1[k,1] = r1a
        radius_1[k,2] = np.median(radius)
        percentages[k,:] = np.percentile(radius,np.linspace(0,100,perc_levels+1))
        
    radius_1 = radius_1*110.0
    percentages = percentages*110.0
    
    if do_plot:
        plt.plot(cluster_sizes,radius_1)
        plt.xlabel('Number of clusters')
        plt.ylabel('Average radius of clusters')
        for i in range(perc_levels):
            if i<perc_levels/2:
                alpha = (i+1)*2.0/perc_levels
            else:
                alpha = (perc_levels-i)*2.0/perc_levels
            plt.fill_between(cluster_sizes,percentages[:,i],percentages[:,i+1],
                alpha=alpha,facecolor='green',edgecolor='none')
    if not chosen_range is None:
        return cluster_sizes[np.argmin(np.abs(radius_1[:,2]-chosen_range))]
def cluster_map(krigor):
    """Visualize distribution spatial distribution of a cluster
    """
    fig = plt.figure(figsize=(7,11))

    Xsel = krigor.X
    
    model = krigor.cluster_results[0]
    n_clusters = model.n_clusters
    cmap = plt.cm.get_cmap("jet",n_clusters)
    
    clu = model.cluster_centers_
    pointsize = np.sqrt(np.bincount(model.labels_))
    
    for i in range(len(Xsel)):
        j = model.labels_[i]
        if (Xsel[i,0]*clu[j,0])<0 and np.abs(np.abs(clu[j,0])-180.0) < 10.0:
            continue
        plt.plot((Xsel[i,0],clu[j,0]),(Xsel[i,1],clu[j,1]),
                 color=cmap(model.labels_[i]),alpha=0.5)
    
    print clu.shape,n_clusters,pointsize.shape
    
    plt.scatter(clu[:,0],clu[:,1],7.5*pointsize,np.linspace(0,n_clusters,n_clusters),'s',
                     alpha=1.0,cmap=cmap,edgecolor='r',linewidth=1.5)
    
    plt.scatter(Xsel[:,0],Xsel[:,1],2,model.labels_,cmap=cmap,alpha=1.0,edgecolor='k')  
    plt.axis('equal')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    #plt.xlim([-90,-20])

# Read in the data
with open('data/canadian_data.csv') as f:
    lines = f.readlines()

# Remove header lines
lines = lines[11:]
    
lon = []
lat = []
moho =[]
topo = []
for i,line in enumerate(lines):
    if line[0] == '\t' or line[0] == '\n' or line[0]==' ' or line[0]=='#':
        continue
    else:
        tokens = line.split(',')
        try:
            lon.append(float(tokens[1]))
            lat.append(float(tokens[2]))
            moho.append(float(tokens[4]))
            topo.append(float(tokens[3]))
        except IndexError:
            print i,tokens
lon = np.asarray(lon)
lat = np.asarray(lat)
moho = np.asarray(moho)
topo = np.asarray(topo)
point_data = np.vstack((lon,lat,moho)).T

plt.scatter(lon,lat,25,moho)
plt.colorbar()
plt.show()



# Define mean and standard deviation of covariance parameters. This is a bit
# convoluted ... but what it basically means is that the 
# Nugget  = 1 km +/- 3 km
# Sill = 6.3 km +/- 6.3 km
# Range = 10 deg +/- 10 deg
# Than this gets converted into the alpha and beta paramter of an inverse 
# gamma-distribution

moments = np.zeros((3,2,1))
moments[:,:,0] = np.array(((1.0,3.0**2),(40.0,40.0**2),(10.0,10.0**2)))
beta = moments[:,0,:]**3/moments[:,1,:]+moments[:,0,:]
alpha = 2 + moments[:,0,:]**2 / moments[:,1,:]

# Test effect of number of clusters on cluster radius
test_cluster_size(np.vstack((lon,lat)).T,20,True)
plt.show()
# Based on this graph, I decide on 10 clusters (1000 km radius)

krigDict = {"hyperPars":np.dstack((alpha,beta)),"minNugget":0.5,"minSill":1.0,
    "maxRange":None,"lambda_w":1000.0,"maxAbsDev":4.0,"maxErrRatio":2.0}
clusterOptions=[{'linkage':'complete','affinity':'precomputed','n_clusters':10}]
krigDict["clusterOptions"] = clusterOptions


# Carry out the clustering and set up the kriging object

cat = np.ones((point_data.shape[0]),dtype=int)

krigor = clean_kriging.MLEKrigor(point_data[:,0],point_data[:,1],point_data[:,2],cat)

krigor._cluster_points(cluster.AgglomerativeClustering,options=clusterOptions,use_pd=True)
krigor._detect_dupes()
krigor._fit_all_clusters(minNugget=krigDict["minNugget"],minSill=krigDict["minSill"],
    hyperpars=np.dstack((alpha,beta)),prior="inv_gamma",maxRange=krigDict["maxRange"])

# Outlier detection

sigma1,new_chosen = krigor.jacknife(krigDict["maxAbsDev"],krigDict["maxErrRatio"],krigDict["lambda_w"])
krigor.chosen_points = new_chosen.copy()
krigor._fit_all_clusters(minNugget=krigDict["minNugget"],minSill=krigDict["minSill"],
    hyperpars=np.dstack((alpha,beta)),prior="inv_gamma",maxRange=krigDict["maxRange"]) 

print 'Outlier detection round 1: %d of %d points selected' % (new_chosen.sum(),len(new_chosen))
print 'Outlier detection round 1: Cross-validation error %.2f km' % np.sqrt(((sigma1[0][:,2]-sigma1[1])**2).mean())
    
sigma2,new_new_chosen =krigor.jacknife(krigDict["maxAbsDev"],krigDict["maxErrRatio"],krigDict["lambda_w"])
krigor.chosen_points = new_new_chosen.copy()
krigor._fit_all_clusters(minNugget=krigDict["minNugget"],minSill=krigDict["minSill"],
    hyperpars=np.dstack((alpha,beta)),prior="inv_gamma",maxRange=krigDict["maxRange"])

print 'Outlier detection round 2: %d of %d points selected' % (new_new_chosen.sum(),len(new_chosen))
print 'Outlier detection round 2: Cross-validation error %.2f km' % np.sqrt(((sigma2[0][:,2]-sigma2[1])**2).mean())


# Prepare the interpolation region
lon = np.arange(np.round(point_data[:,0].min()),np.round(point_data[:,0].max()+1),1)
lat = np.arange(np.round(point_data[:,1].min()),np.round(point_data[:,1].max()+1),1)

lonGrid,latGrid = np.meshgrid(lon,lat)
cat_grid = np.ones(lonGrid.shape,dtype=int)

# Carry out the interpolation
pred,krigvar,predPars = krigor.predict(lonGrid.flatten(),latGrid.flatten(),cat_grid.flatten(),
                                       lambda_w=krigDict["lambda_w"],get_covar=False)

pred = pred.reshape(lonGrid.shape)
krigvar = krigvar.reshape(lonGrid.shape)

# Plot the interpolation result and associated uncertainty
plt.contourf(lon,lat,pred)
plt.colorbar()

plt.figure()
plt.contourf(lon,lat,np.sqrt(krigvar))
plt.colorbar()

header=u"""Depth to Moho boundary obtained by global non-stationary kriging of data from Geological survey of Canada
===
Author
===
Wolfgang Szwillus, Kiel University
wolfgang.szwillus@ifg.uni-kiel.de
===
Description
===
Column 1: Longitude
Column 2: Latitude
Column 3: Depth to Moho (km), measured from 0
Column 4: Estimated uncertainty of Moho depth (km)
=== 
Method
===
See Szwillus et al. (2019):  https://doi.org/10.1029/2018JB016593
===
Parameters
===
"""
now = datetime.datetime.now()
temp = np.vstack((lonGrid.flat,latGrid.flat,pred.flat,np.sqrt(krigvar).flat)).T
np.savetxt("Canada-moho-interp-%s.txt"%now.strftime("%y-%B-%d-%H-%M"),temp,header=unicode(header),fmt='%.2f')