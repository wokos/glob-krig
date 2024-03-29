import numpy as np
from scipy.optimize import minimize

from collections import defaultdict
import scipy.sparse.linalg as splinalg
from scipy.sparse import csc_matrix,lil_matrix
from .func_dump import *

def meta_kriging(pointData,predictionData,cluster_settings,optimization_settings,
                jackknife_settings=None,more_returns=False,get_covar=False,blocks=1,verbose=False):
    """Convenience interface to Maximum Likelihood Kriging

    Parameters
    ----------
        PointData: np.ndarray
            Contains columns (lon,lat,vals,cat) of input data. `vals` are the values to interpolate,
            `cat` is the category.
        predictionData: (lon,lat,cat)
            Contains locations at which to evaluate the interpolation.
        cluster_settings : dict
            Contains the following parameters for controlling how the points will be clustered
            constructorFunc : function
                This is a constructor function for a clustering predictor. Look at documentation for
                scikit-learn for examples.
            cluster_options : dict
                Dictionary of additional arguments that will be passed to `constructorFunc`
            use_spherical_distance : bool
                If true, clustering will be based on geographical distances, otherwise `constructorFunc` will
                calculate them using cartesian coordinates
            threshold : int
                Clusters containing less than `threshold` points will be removed.
        optimization_setings : dict
            Contains the following paramters for controlling how the values of (nugget,sill,range) will be determined
            lambda_w : float
                Parameter controlling the interpolation behaviour of covariance parameters (sill, nugget, range)
                between cluster centers.
            minNugget : float, optional
                Minimum value for the nugget parameter. Note that it is given as a variance, so if you want
                a minimum nugget of 10 km, you would have to put 100 km**2 into this function.
            minSill : float, optional
                Minimum value for the sill parameter. Note that it is also given as a variance!
            maxRange : float, optional
                Maximum value for the correlation range in *degrees*!
            hyperPars: np.ndarray
                Parameters describing the prior of the covariance parameters (nugget, sill range).
            prior : str
                Which kind of prior to use for the covariance parameters. Typically should be "inv_gamma".
            badPoints : np.ndarray
                Points where `badPoints` is True, will not be used for interpolation. However, they are used for
                defining clusters.                
        jackknife_settings : dict,optional
            Contains parameters for controlling the behaviour of the jackknife outlier selection. If this is None,
            no jackknife outlier selection will be performed.
            maxAbsError : float
                Maximum acceptable error for jacknifing.
            maxRelError : float
                Maximum relative error for jacknifing.
        get_covar : bool
            If True, the function will return not only the variance (i.e. uncertainty) at each location, but also
            the full covariance matrix. This can quickly become extremely numerically expensive, when the number 
            of prediction points is large.
        blocks : int
            If larger than one,  the prediction data set will be divided into blocks. This reduces memory
            consumption. Can only be used if `get_covar` is False.
        verbose : bool,optional
            Control the level of console spam produced by this function.
    """
    constructorFunc = cluster_settings.get("constructorFunc",None)
    cluster_options = cluster_settings.get("cluster_options",None)
    use_spherical_distance = cluster_settings.get("use_spherical_distance",False)
    threshold = cluster_settings.get("threshold",10)
    
    lambda_w = optimization_settings.get("lambda_w",100.0)
    minNugget = optimization_settings.get("minNugget",None)
    minSill = optimization_settings.get("minSill",None)
    maxRange = optimization_settings.get("maxRange",None)
    badPoints = optimization_settings.get("badPoints",None)
    if badPoints is None:
        badPoints = np.zeros((pointData[0].shape),dtype=bool)
    hyperPars = optimization_settings.get("hyperPars",None)
    prior = optimization_settings.get("prior",None)    

    
    
    pred = np.ones(predictionData[0].shape) * np.nan
    krigvar = np.ones(predictionData[0].shape) * np.nan

    krigor = MLEKrigor(pointData[0],pointData[1],pointData[2],pointData[3])
    krigor._cluster_points(constructorFunc,options=cluster_options,use_pd=use_spherical_distance)
    krigor._detect_dupes()
    krigor.chosen_points[badPoints] = 0
    krigor._fit_all_clusters(minNugget=minNugget,minSill=minSill,
        hyperpars=hyperPars,prior=prior,maxRange=maxRange)
    krigor._reassign_small_clusters(threshold=threshold)
    if not jackknife_settings is None:
        maxAbsError = jackknife_settings.get("maxAbsError",None)
        maxRelError = jackknife_settings.get("maxRelError",2.0)

        sigma1,new_chosen = krigor.jacknife(maxAbsError,maxRelError,lambda_w)
        new_chosen[badPoints] = 0
        krigor.chosen_points = new_chosen.copy()
        krigor._fit_all_clusters(minNugget=minNugget,minSill=minSill,
            hyperpars=hyperPars,prior=prior,maxRange=maxRange)        
        krigor._reassign_small_clusters(threshold=threshold)

        sigma2,new_new_chosen = krigor.jacknife(maxAbsError,maxRelError,lambda_w)
        new_new_chosen[badPoints] = 0
        krigor.chosen_points = new_new_chosen.copy()
        krigor._fit_all_clusters(minNugget=minNugget,minSill=minSill,
            hyperpars=hyperPars,prior=prior,maxRange=maxRange)
        
    if get_covar:
        pred,krigvar,_ = krigor.predict(predictionData[0].flatten(),predictionData[1].flatten(),predictionData[2].flatten(),lambda_w=lambda_w,get_covar=get_covar)
        pred = pred.reshape(predictionData[0].shape)
    else:
        Npred = len(predictionData[0].flatten())
        block_ixs = np.array_split(range(Npred),blocks)
        pred = np.ones((Npred)) * np.nan        
        krigvar = np.ones((Npred)) * np.nan
        for _,block_ix in enumerate(block_ixs):
            xpred = predictionData[0].flatten()[block_ix]
            ypred = predictionData[1].flatten()[block_ix]
            catpred = predictionData[2].flatten()[block_ix]
            pred[block_ix],krigvar[block_ix],_ =  krigor.predict(
                xpred,ypred,catpred,lambda_w=lambda_w,get_covar=get_covar)
        pred = pred.reshape(predictionData[0].shape)
        krigvar = krigvar.reshape(predictionData[0].shape)

    if more_returns and not jackknife_settings is None:
        return pred,krigvar,krigor,sigma1,sigma2
    else:
        return pred,krigvar,krigor


def log_inv_gamma(vals,hyperpars):
    """Calculate the logpdf of inverse gamma distribution for cov parameters
    """
    return np.sum(-(hyperpars[:,0]+1)*np.log(vals) - hyperpars[:,1]/vals)

def log_gamma(vals,hyperpars):
    """Calculate the logpdf of gamma distribution for cov parameters
    """
    return np.sum((hyperpars[:,0]-1)*np.log(vals) - vals/hyperpars[:,1])

def log_gauss(vals,hyperpars):
    """Calculate logpdf of normal distribution (mean,sigma)
    """
    return np.sum((vals-hyperpars[:,0])**2/hyperpars[:,1]**2)

def spherical_average(lon,lat):
    """Calculate spherical average of the points (lon,lat) via 3d space
    """
    theta = (90-lat)/180.0*np.pi
    phi = lon / 180.0 * np.pi
    x = np.sin(theta)*np.cos(phi)
    y = np.sin(theta)*np.sin(phi)
    z = np.cos(theta)
    
    xc,yc,zc = x.mean(),y.mean(),z.mean()
    
    thetac = np.arctan2(np.sqrt(xc**2+yc**2),zc)
    phic = np.arctan2(yc,xc)
    lonc = 180.0/np.pi * phic
    latc = (90 - 180.0/np.pi*thetac)
    return lonc,latc

def C_sph_nugget(d,covarParams):
    # Nugget,sill,range
    C = np.zeros(d.shape)
    if covarParams[2] == 0.0:
        np.fill_diagonal(C,covarParams[0] + covarParams[1])
    else:
        C[d==0] = covarParams[0] + covarParams[1]
        C[(d>0)&(d<covarParams[2])] =  covarParams[1] * (1 - 1.5*d[(d>0)&(d<covarParams[2])]/covarParams[2] + 0.5 * d[(d>0)&(d<covarParams[2])]**3  / covarParams[2]**3)
        C[(d>0)&(d>=covarParams[2])] =  0.0
    return C

def interp_pars(x,y,cluster_x,cluster_y,all_pars,lambda_w = 100.0):
    """Interpolate cov parameters between cluster centers
    See Risser and Calder 2017, section 3.1.
    """
    valid = (~np.isnan(cluster_x)) & (~np.isnan(cluster_y))
    cpd = get_pairwise_cross_distance(x,y,np.array(cluster_x),np.array(cluster_y))
    raw_weights = np.exp(-cpd**2/(2.0*lambda_w))
    weights = raw_weights[:,valid] / raw_weights[:,valid].sum(1)[:,None]
    interpolated_pars = weights.dot(all_pars[valid,:])
    return interpolated_pars    

def stationary_likelihood_func(pd,Z,pars,covar_func=C_sph_nugget,covariates=None):
    """Evaluate mvn derived from covariance function
    
    For numerical stabilization, I first try to calculate the cholesky 
    decomposition of the covariance matrix. If this fails, an eigenvalue 
    decomposition is used next, and all negative eigenvalues are pushed up in
    order to ensure a positive definite matrix.
    """

    sigma = covar_func(pd,pars)
    try:
        L = np.linalg.cholesky(sigma)
    except np.linalg.LinAlgError:
        sigmaOld = sigma.copy()
        v,Q = np.linalg.eig(sigma)
        v[v<0] = v[v<0] - v.min() + 1e-4 * (v.max() - v.min())
        sigma = Q.dot(np.diag(v)).dot(np.linalg.inv(Q))
        L = np.linalg.cholesky(sigma)
    mu = Z.mean()
    logdet = 2 * (np.log(L.diagonal())).sum()
    Linv = np.linalg.inv(L)
    sigmaInv = Linv.T.dot(Linv)
    if covariates is None:
        return -0.5 * (logdet + (Z-mu).T.dot(sigmaInv).dot((Z-mu)))
    else:
        # See Risser and Calder 2015 4.1 Local likelihood estimation
        A = np.linalg.inv(covariates.T.dot(sigmaInv.dot(covariates)))
        B = covariates.dot(A).dot(covariates.T).dot(sigmaInv)
        P = sigmaInv.dot(np.eye(B.shape[0])-B)
        Plogdet = np.linalg.slogdet(P)[1]
        return -0.5 * (logdet + Plogdet + Z.T.dot(P).dot(Z))
    
def MLE_radius_bayes(x,y,vals,x0,hyperpars,radius=10.0,minSill=0.0,
                    minNugget=0.0,maxRange=180.0,prior="inv_gamma",covariates=None):
    """Bayesian MLE estimate of covariance parameters (sill, nugget, range) 
    
    Parameters
    ----------
    x : np.array
        Longitudes
    y : np.array
        Latitudes
    vals: np.array
        Values
    x0: tuple
        Longitude, latitude of center
    hyperpars: np.array
        Hyperparameters describing the prior put on the covariace parameters.
        The array should be shape (3,n).  hyperpars[0,:] are the parameters 
        for the nugget. This could be extended in the future, if you want to
        use more than 3 parameters for the covariance function, but at the 
        moment the spherical nugget function with 3 parameters is hard-coded in.
    covariates: np.array
        Allows for correct estimation, if a trend has been subtracted.
        I don't use this at the moment.
    """
    
    d0 = get_all_geo_distance(x,y,x0[0],x0[1])
    if radius is None:
        in_circle = np.ones((len(x)),dtype=bool)
    else:
        in_circle = d0<=radius
    valVar = vals[in_circle].var()
    pd = get_pairwise_geo_distance(x[in_circle],y[in_circle])
    if not covariates is None:
        covariates = covariates[in_circle]
    
    if not isinstance(prior,str):
        func = lambda theta:-stationary_likelihood_func(pd,vals[in_circle],theta,covariates=covariates)-prior(theta,hyperpars)
    elif prior == "inv_gamma":
        func = lambda theta:-stationary_likelihood_func(pd,vals[in_circle],theta,covariates=covariates)-log_inv_gamma(theta,hyperpars)
    elif prior =="gamma":
        func = lambda theta:-stationary_likelihood_func(pd,vals[in_circle],theta,covariates=covariates)-log_gamma(theta,hyperpars)
    elif prior =="normal":
        func = lambda theta:-stationary_likelihood_func(pd,vals[in_circle],theta,covariates=covariates)-log_gauss(theta,hyperpars)
        
    if len(x) == 0 or pd.max()==0:
        print("Not enough points")
        if prior == "inv_gamma":
            return hyperpars[:,1]/(hyperpars[:,0]-1)
        elif prior == "gamma":
            return hyperpars[:,0]*hyperpars[:,1]
    
    optireturn = minimize(func,[0.1*valVar,0.9*valVar,0.9*pd.max()],
                      options={"maxiter":100},
                      method='L-BFGS-B',bounds=((minNugget,None),(minSill,None),(0.0,maxRange)))
    return optireturn.x

def MLE_radius(x,y,vals,x0,radius=10.0,minSill=0.0,minNugget=0.0,use_lims=True,covariates=None):
    d0 = get_all_geo_distance(x,y,x0[0],x0[1])
    if radius is None:
        radius = d0.max()
    in_circle = d0<=radius
    pd = get_pairwise_geo_distance(x[in_circle],y[in_circle])
    func = lambda theta:-stationary_likelihood_func(pd,
                                                   vals[in_circle],theta,covariates=covariates)
    if use_lims:
        sillMax = max(vals.var(),minSill)
        nuggetMax = max(minNugget,vals.var())
        rangeMax = pd.max()
    else:
        sillMax = None
        nuggetMax = None
        rangeMax = None
    optireturn = minimize(func,[0.1*vals.var(),0.9*vals.var(),0.9*pd.max()],
                      options={"maxiter":100},
                      method='L-BFGS-B',bounds=((minNugget,nuggetMax),(minSill,sillMax),(0.0,rangeMax)))
    return optireturn.x

def memory_saver_C_sph_nugget_ns(x,y,pars,nblocks=10):
    """Memory efficient method of calculating cov matrix from cov function
    
    A spherical covariance function is used. This function is exactly zero, if
    two points are more separated than their effective range. To save memory
    a sparse matrix representation is constructed by splitting the points in
    blocks and calculating only blockwise distance matrices.
    Inherently, there is a trade-off between the memory reduction and 
    CPU increase. 
    """
    N = len(x)
    cut_indices = np.array_split(range(0,N,1),nblocks)
    C = lil_matrix((N,N))
    for i in range(nblocks):
        row_indices = cut_indices[i]
        block_pd = get_pairwise_cross_distance(x[row_indices],y[row_indices],x,y)
        block_pd[block_pd<1e-5] = 0
        rhoEff = np.sqrt(2) * pars[:,2] * pars[row_indices,2,None] / np.sqrt(pars[:,2]**2+pars[row_indices,2,None]**2)
        sigmaEff = np.sqrt(pars[:,1]*pars[row_indices,1,None])
        normd = block_pd/rhoEff
        block_C = sigmaEff * rhoEff * ( 1-1.5*normd+0.5*normd**3) / np.sqrt(pars[:,2]*pars[row_indices,2,None])
        block_C[normd>1] = 0.0
        nuggetEff = 0.5  * (pars[:,0] + pars[row_indices,0,None])
        block_C[block_pd==0] = block_C[block_pd==0] + nuggetEff[block_pd==0]
        C[row_indices,:] = C[row_indices,:] + block_C
    return C

class MLEKrigor:
    """Maximum Likelihood Estimate Kriging with non-stationary cov function
        
    Note
    ----
    Based on Risser and Calder (2017), https://arxiv.org/abs/1507.08613v4
    """


    def __init__(self,x,y,vals,cat=None,covariates=None):
        """
        Parameters
        ----------
        x,y,vals : np.array
            lon,lat and value of point data to interpolate
        cat : np.array
            Optionally, gives a category for each point. All categories
            are treated independently. dtype=int
        covariates: np.array
            Optionally, give a matrix of covariates (C), such that 
            $y = C \beta$
        """
        self.X = np.zeros((len(x),3))
        self.X[:,0] = x
        self.X[:,1] = y
        self.X[:,2] = vals
        if cat is None:
            self.cat = np.ones((len(x)),dtype=int)
        else:
            self.cat = cat
        self.covariates = covariates        
        self.allCats = np.unique(self.cat)
        self._detect_dupes()
        self.chosen_points = ~self.is_dupe
        
    def _cluster_points(self,constructorFunc,options={'bandwidth':10},use_pd=False,
            exclude_points=None):
        """Use scikit-learn functions to cluster the points
        
        Separate clustering objects are created for each category and are stored
        in self.cluster_results
        
        Parameters
        ----------
        constructorFunc: function
            This function initializes a clustering object from scikit-learn
        options : dict or list of dict
            These options are passed to constructFunc. Different options can be
            given for different categories (then, options is a list of dicts)
        use_pd : bool
            Some clustering algorithms need only the pairwise distance matrix.
            If true the pd matrix will be passed to the constructorFunc
            If false the actual point locations will be used instead.
        exclude_points: np.array, optional
            Needs to be dtype=bool
            If given, the specified points will be excluded from clustering,
            otherwise all points are used.
        """
        self.cluster_results = []
        if exclude_points is None:
            sel = np.ones((self.X.shape[0]),dtype=bool)
        else:
            sel = ~exclude_points
            
        if not type(options)==list:
            options = cycle([options])
        for i,(c,opts) in enumerate(zip(self.allCats,options)):
            clusterer = constructorFunc(**opts)
            if use_pd:
                pd = get_pairwise_geo_distance(self.X[(self.cat==c)&sel,0],self.X[(self.cat==c)&sel,1])
                clusterer.fit(pd)
            else:
                clusterer.fit(self.X[(self.cat==c)&sel,0:2])
            self.cluster_results.append(clusterer)
            
    def _detect_dupes(self):
        """Detect and mark points which are at the same geographical location
        """
        unique_xy = defaultdict(list)
        for i in range(len(self.X)):
            unique_xy[self.X[i,0],self.X[i,1]].append(i)
        is_dupe = np.ones((len(self.X)),dtype=bool)
        for u in unique_xy:
            if len(unique_xy[u])==1:
                is_dupe[unique_xy[u]] = False
        self.is_dupe = is_dupe
        self.unique_xy = unique_xy
    
    def _fit_all_clusters(self,minNugget=0.0,minSill=0.0,maxRange=None,hyperpars=None,prior=None):
        """Fit local cov parameters (nugget,sill,range) for each cluster
        
        Uses Local likelihood estimation
        """
        self.allPars = []
        if not hyperpars is None:
            assert hyperpars.shape == (3,len(np.unique(self.allCats)),2)

        for i,c in enumerate(self.allCats):
            selChosen = self.chosen_points[self.cat==c]
            Xsel = self.X[self.cat==c,:]
            ms = self.cluster_results[i]
            labels = ms.labels_
            labels_unique = np.unique(labels)
            n_clusters_ = len(labels_unique)
            
            if hasattr(ms,'cluster_centers_'):
                cluster_centers = ms.cluster_centers_
            else:
                cluster_centers = np.zeros((n_clusters_,2))
                for k in range(n_clusters_):
                    lonc,latc = spherical_average(Xsel[(labels==k)&(selChosen),0],Xsel[(labels==k)&(selChosen),1])
                    cluster_centers[k,0] = lonc
                    cluster_centers[k,1] = latc
                ms.cluster_centers_ = cluster_centers.copy()

            all_pars = np.zeros((n_clusters_,3))
            for k in range(n_clusters_):
                my_members = labels == k  
                cluster_center = cluster_centers[k]
                if np.sum(my_members)<1:
                    all_pars[k,:] = (0.0,0.0,1.0)
                if hyperpars is None:
                    all_pars[k,:] = MLE_radius(Xsel[my_members & selChosen,0],Xsel[my_members& selChosen,1],
                                  Xsel[my_members& selChosen,2],cluster_center,
                                  radius=None,minNugget=minNugget,minSill=minSill,covariates=self.covariates)
                else:
                     all_pars[k,:] = MLE_radius_bayes(Xsel[my_members & selChosen,0],Xsel[my_members& selChosen,1],
                  Xsel[my_members& selChosen,2],cluster_center,hyperpars[:,i,:],
                  radius=None,minNugget=minNugget,minSill=minSill,prior=prior,maxRange=maxRange,covariates=self.covariates)
            self.allPars.append(all_pars)
            
    
    def _reassign_small_clusters(self,threshold=10):
        for i,c in enumerate(self.allCats):
            ms = self.cluster_results[i]
            catChosen = self.chosen_points[self.cat==c]
            cluster_centers = ms.cluster_centers_
            labels = ms.labels_
            labels_unique = np.unique(labels)
            n_clusters_ = len(labels_unique)
            cluster_sizes = np.array([len(np.where((catChosen) & (labels==k))[0]) for k in range(n_clusters_)])
            big_clusters = cluster_sizes > threshold
            if np.sum(big_clusters)<1:
                continue
            cluster_x = np.array([c[0] for c in cluster_centers])
            cluster_y = np.array([c[1] for c in cluster_centers])
            cluster_pd = get_pairwise_geo_distance(cluster_x,cluster_y)
            for k in range(n_clusters_):
                if cluster_sizes[k]>=threshold:
                    continue
                sort_indices = np.argsort(cluster_pd[k,:])
                sorted_big_clusters = big_clusters[sort_indices]
                nearest_neighbor = sort_indices[np.where(sorted_big_clusters)[0][0]]
                assert not cluster_pd[k,nearest_neighbor] == 0
                self.allPars[i][k,:] = self.allPars[i][nearest_neighbor,:]

    def _interp_pars_to_points(self,lambda_w=100.0):
        self.pointPars = np.zeros((len(self.X),3))
        for i,c in enumerate(self.allCats):
            ms = self.cluster_results[i]
            cluster_centers = ms.cluster_centers_
            labels = ms.labels_
            labels_unique = np.unique(labels)
            n_clusters_ = len(labels_unique)
            cluster_x = np.array([clu[0] for clu in cluster_centers])
            cluster_y = np.array([clu[1] for clu in cluster_centers])
            Xsel = self.X[self.cat==c,:]
            self.pointPars[self.cat==c,:] = interp_pars(Xsel[:,0],Xsel[:,1],cluster_x,cluster_y,self.allPars[i])
            
    def predict(self,lonPred,latPred,catPred,lambda_w=100.0,get_covar=True):
        predPars = np.zeros((lonPred.shape[0],3))
        predicted = np.zeros(lonPred.shape)
        if get_covar:
            predSigma = lil_matrix((lonPred.shape[0],lonPred.shape[0]))
        else:
            predSigma = np.zeros((lonPred.shape[0]))
        for i,c in enumerate(self.allCats):
            Xsel = self.X[(self.cat==c) & self.chosen_points,:]
            ms = self.cluster_results[i]
            cluster_centers = ms.cluster_centers_
            labels = ms.labels_
            labels_unique = np.unique(labels)
            n_clusters_ = len(labels_unique)
            cluster_x = np.array([clu[0] for clu in cluster_centers])
            cluster_y = np.array([clu[1] for clu in cluster_centers])
            
            lonSel = lonPred[catPred==c]
            latSel = latPred[catPred==c]
            Y = np.zeros((len(lonSel),2))
            Y[:,0] = lonSel
            Y[:,1] = latSel
            
            print("Solving kriging system for category %d with no. points %d %d " %(c,len(lonSel),len(Xsel)))
            temp = solve_kriging_system(Xsel,Y,cluster_x,cluster_y,self.allPars[i],
                                        lambda_w=lambda_w,get_covar=get_covar)
            pred_ix = np.where(catPred==c)[0]
            predicted[catPred==c] = temp[0]
            if get_covar:
                predSigma[np.ix_(pred_ix,pred_ix)] = temp[1] 
            else:
                predSigma[catPred==c] = temp[1]
            predPars[catPred==c,:] = temp[2]
        return predicted,predSigma,predPars
        
    def jacknife(self,maxAbsDev=5.0,maxErrRatio=2.0,lambda_w=100.0,verbose=False):
        """
        Use only points for prediction which are in the same cluster
        """
        jpred = np.zeros((self.X.shape[0]))
        krigvar = np.zeros((self.X.shape[0]))
        for i,c in enumerate(self.allCats):
            Xcat = self.X[self.cat==c,:]
            ms = self.cluster_results[i]
            cluster_centers = ms.cluster_centers_
            labels = ms.labels_
            labels_unique = np.unique(labels)
            n_clusters_ = len(labels_unique)
            cluster_x = np.array([clu[0] for clu in cluster_centers])
            cluster_y = np.array([clu[1] for clu in cluster_centers])
            jpred_cat = np.zeros((Xcat.shape[0]))
            krigvar_cat = np.zeros((Xcat.shape[0]))
            for k,label in enumerate(labels_unique):
                Xsel = Xcat[labels==label,:]
                chosen_points_sel = self.chosen_points[self.cat==c][labels==label]
                if Xsel.shape[0]<3:
                    jpred_cat[labels==label] = Xsel[:,2].mean()
                    krigvar_cat[labels==label] = 0.0
                    continue
                temp1,temp2 = jacknife_kriging(Xsel,chosen_points_sel,
                                             np.asarray(cluster_x),
                                               np.asarray(cluster_y),self.allPars[i],lambda_w,verbose=verbose)
                
                jpred_cat[labels==label] = temp1
                krigvar_cat[labels==label] = temp2
            jpred[self.cat==c] = jpred_cat
            krigvar[self.cat==c] = krigvar_cat
        returnor = [self.X,jpred,krigvar]
        absDev = np.abs(jpred - self.X[:,2])
        ratDev = absDev / np.sqrt(krigvar) 
        ok = (absDev <= maxAbsDev) | (ratDev <= maxErrRatio)
        # All points which are not duplicates become chosen if they fulfil the conditions
        new_chosen_points = np.zeros(self.chosen_points.shape,dtype=bool)
        new_chosen_points[ok&(~self.is_dupe)] = True
        # Duplicate selection
        # All points which are duplicates are chosen if they fulfil the conditions AND they
        # have the lowest prediction error of all points at the same position
        for u in self.unique_xy:
            if len(self.unique_xy[u]) == 1:
                continue

            dupErrs = absDev[self.unique_xy[u]]
            winner = self.unique_xy[u][np.argmin(dupErrs)]
            
            if verbose:
                print("Duplicate indices",self.unique_xy[u])
                print("Errors",dupErrs)
                print("Winner",winner,ratDev[winner])

            if (ratDev[winner]<=maxErrRatio) | (absDev[winner] <= maxAbsDev):
                new_chosen_points[winner] = True
        
        return returnor,new_chosen_points
            

def solve_kriging_system(X,Y,cluster_x,cluster_y,allPars,lambda_w=100.0,get_covar=True):
    lonSel = Y[:,0]
    latSel = Y[:,1]
    combX = np.hstack((Y[:,0],X[:,0]))
    combY = np.hstack((Y[:,1],X[:,1]))
    combPars = interp_pars(combX,combY,cluster_x,cluster_y,allPars,lambda_w=lambda_w)
    nblocks = 1
    if len(combX)>1000:
        nblocks = 10
    bigSigma = memory_saver_C_sph_nugget_ns(combX,combY,combPars,nblocks=nblocks)
    Npoint = len(X)
    Nsel = len(Y)
    gen1 = range(Nsel,Nsel+Npoint)  # Z
    gen2 = range(Nsel) # Z*
    pointSigma = bigSigma[np.ix_(gen1,gen1)]
    crossSigma = bigSigma[np.ix_(gen2,gen1)]
    selSigma = bigSigma[np.ix_(gen2,gen2)]
    pointSigma = csc_matrix(pointSigma)
    mu = X[:,2].mean()
    phi = splinalg.gmres(pointSigma,X[:,2]-mu,tol=1.0e-4)
    
    crossSigma = csc_matrix(crossSigma)
    predicted = mu + crossSigma.dot(phi[0])
    
    psi = np.zeros((Npoint,Nsel))
    for k in range(Nsel):
        A = pointSigma
        b = crossSigma[k,:].toarray().T
        temp  = splinalg.gmres(A,b,tol=0.1)
        psi[:,k] = temp[0]
    if get_covar:
        oerk = crossSigma.dot(psi)
        oerk = csc_matrix(oerk)
        predSigma = (selSigma - oerk).toarray()
    else:
        predSigma = selSigma.diagonal() - np.sum(crossSigma.toarray()*psi.T,1)
    return predicted,predSigma,combPars[:Nsel,:]

    
def solve_stationary_kriging_system(X,Y,C_func,pars,get_covar=True,n_blocks=1):
    Npoint = X.shape[0]
    Nsel = Y.shape[0]
    
    lonSel = Y[:,0]
    latSel = Y[:,1]
    combX = np.hstack((Y[:,0],X[:,0]))
    combY = np.hstack((Y[:,1],X[:,1]))

    bigSigma = lil_matrix((combX.size,combX.size))
    ixs = np.array_split(np.arange(combX.size,dtype=int),n_blocks)
    for i in range(n_blocks):
        ix = ixs[i]
        block_pd = get_pairwise_cross_distance(combX[ix],combY[ix],combX,combY)
        block_pd[block_pd<1e-5] = 0
        block_C = C_func(block_pd,pars)
        print(bigSigma.shape,block_C.shape)
        bigSigma[ix,:] = bigSigma[ix,:] + block_C
        
    gen1 = range(Nsel,Nsel+Npoint)  # Z
    gen2 = range(Nsel) # Z*
    pointSigma = bigSigma[np.ix_(gen1,gen1)]
    crossSigma = bigSigma[np.ix_(gen2,gen1)]
    selSigma = bigSigma[np.ix_(gen2,gen2)]
    pointSigma = csc_matrix(pointSigma)
    mu = X[:,2].mean()
    phi = splinalg.gmres(pointSigma,X[:,2]-mu,tol=1.0e-4)
    
    crossSigma = csc_matrix(crossSigma)
    predicted = mu + crossSigma.dot(phi[0])
    
    psi = np.zeros((Npoint,Nsel))
    for k in range(Nsel):
        A = pointSigma
        b = crossSigma[k,:].toarray().T
        temp  = splinalg.gmres(A,b,tol=0.1)
        psi[:,k] = temp[0]
    if get_covar:
        oerk = crossSigma.dot(psi)
        oerk = sparse.csc_matrix(oerk)
        predSigma = (selSigma - oerk).toarray()
    else:
        predSigma = selSigma.diagonal() - np.sum(crossSigma.toarray()*psi.T,1)
    return predicted,predSigma
    
    
def jacknife_kriging_all_chosen(X,cluster_x,cluster_y,allPars,lambda_w=100.0):
    if len(X)<=1:
        print("Jacknife: Not enough points")
        return X[:,2],np.ones((1))*np.inf
    combPars = interp_pars(X[:,0],X[:,1],cluster_x,cluster_y,allPars,lambda_w=lambda_w)
    bigSigma = memory_saver_C_sph_nugget_ns(X[:,0],X[:,1],combPars,nblocks=1)
    mu = X[:,2].mean()
    predicted = np.zeros((len(X)))
    phiAll = splinalg.gmres(bigSigma,X[:,2]-mu)[0]
    krigvar = np.zeros((len(X)))
    Npoints = X.shape[0]
    for k in range(len(X)):
        rowsAll = [i for i in range(Npoints) if not i==k]        
        A = bigSigma[np.ix_(rowsAll,rowsAll)]
        #phi = phiAll[rowsAll]
        phi = splinalg.gmres(A,X[rowsAll,2]-X[rowsAll,2].mean())[0]
        crossSigma = bigSigma[k,rowsAll]
        crossSigma = csc_matrix(crossSigma)
        predicted[k] = X[rowsAll,2].mean() + crossSigma.dot(phi)
        rhs = crossSigma.toarray().T
        psi = splinalg.gmres(A,rhs,tol=0.1)[0]
        krigvar[k] =  bigSigma[k,k] - np.inner(crossSigma.toarray(),psi)[0]
    return predicted,krigvar


def jacknife_kriging(X,chosenPoints,cluster_x,cluster_y,allPars,lambda_w=100.0,verbose=False):
    predicted = np.ones((X.shape[0]))*np.nan
    krigvar = np.ones((X.shape[0]))*np.inf
    # Actual leave-one-out jacknifing
    temp = jacknife_kriging_all_chosen(X[chosenPoints,:],cluster_x,cluster_y,allPars,lambda_w=lambda_w)
    if verbose:
        print('chosen',np.sum(chosenPoints),np.sum(~chosenPoints),temp[0].shape)
    predicted[chosenPoints] = temp[0]
    krigvar[chosenPoints] = temp[1]
    # Non-chosen points are simply predicted using chosen points
    if np.sum(~chosenPoints) > 0 and np.sum(chosenPoints) > 0:
        temp = solve_kriging_system(X[chosenPoints,:],X[~chosenPoints,:],cluster_x,cluster_y,allPars,lambda_w=lambda_w,get_covar=False)
        if verbose:
            print('not chosen',np.sum(chosenPoints),np.sum(~chosenPoints),temp[0].shape,temp[1].shape)

        predicted[~chosenPoints] = temp[0]
        krigvar[~chosenPoints] = temp[1]
    return predicted,krigvar