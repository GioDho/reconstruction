#!/usr/bin/env python
import numpy as np
from skimage.measure import LineModelND, ransac
from scipy import spatial
from clusterTools import Cluster

def array_row_intersection(a,b):
   tmp=np.prod(np.swapaxes(a[:,:,None],1,2)==b,axis=2)
   return a[np.sum(np.cumsum(tmp,axis=0)*tmp==1,axis=1).astype(bool)]

class ClusterMatcher:
    def __init__(self,params):
        self.min_length = params['min_length']
        self.npixx = params['npixx']
        self.min_pix_intercept = params['min_npix_intercept']
        self.min_samples_ransac = params['min_samples_ransac']
        self.residual_threshold_ransac = params['residual_threshold_ransac']
        self.max_trials_ransac = params['max_trials_ransac']
        
    def fitCluster(self,hits,plotting=False):
        #print (hits)
        x = np.array([h[0] for h in hits])
        y = np.array([h[1] for h in hits])
        data = np.column_stack([x, y])
        
        # fit line using all data
        model = LineModelND()
        model.estimate(data)

        # robustly fit line only using inlier data with RANSAC algorithm
        model_robust, inliers = ransac(data, LineModelND, min_samples=self.min_samples_ransac,
                                       residual_threshold=self.residual_threshold_ransac, max_trials=self.max_trials_ransac)
        outliers = inliers == False

        line_x = np.arange(0, self.npixx)
        line_y = model.predict_y(line_x)
        line_y_robust = model_robust.predict_y(line_x)

        if plotting:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.plot(data[inliers, 0], data[inliers, 1], '.b', alpha=0.6,
                    label='Inlier data')
            ax.plot(data[outliers, 0], data[outliers, 1], '.r', alpha=0.6,
                    label='Outlier data')
            ax.plot(line_x, line_y, '-k', label='Line model from all data')
            ax.plot(line_x, line_y_robust, '-b', label='Robust line model')
            ax.set_ylim(0,self.npixx)
            ax.legend(loc='lower left')
            plt.show()

        extrap_xy = np.column_stack([line_x, line_y]).astype(int)
        extrap_xy_robust = np.column_stack([line_x, line_y_robust]).astype(int)
        return extrap_xy,extrap_xy_robust

    def matchClusters(self,killer,targets):
       # do the fit on the ZS hits to find better the axis wrt the outliers
       if killer.shapes['long_width'] > self.min_length:
            killer_x = np.array([h[0] for h in killer.hits_fr_zs])
            killer_y = np.array([h[1] for h in killer.hits_fr_zs])
            killer_data = np.column_stack([killer_x, killer_y])
            extrap_xy,extrap_xy_robust = self.fitCluster(killer.hits_fr_zs)
            #print ("KILLER extrap")
            #print(extrap_xy)
            for i,clu in enumerate(targets):
                hits = clu.hits
                x = np.array([h[0] for h in clu.hits_fr_zs])
                y = np.array([h[1] for h in clu.hits_fr_zs])
                data = np.column_stack([x, y])
                #print ("data cluster = ",i)
                #print (data)
                intersect = array_row_intersection(extrap_xy,data)
                intersect_robust = array_row_intersection(extrap_xy_robust,data)
                #print ('INTERSECT = ')
                #print (intersect)
                
                if max(len(intersect),len(intersect_robust))>self.min_pix_intercept:
                    # This solution is optimal when data is very large
                    tree = spatial.cKDTree(data)
                    mindist, minid = tree.query(killer_data)
                    #print("Min dist = ",min(mindist))
                    clu.minDistKiller = min(mindist)
                    clu.nMatchKiller = len(intersect_robust)
                    clu.nMatchKillerWeak = len(intersect)

        
if __name__ == '__main__':

   cosm_cand = 2
   hits = np.load('debug_code/sclu_{cand}.npy'.format(cand=cosm_cand))

   fileParModule = open('modules_config/clusterMatcher.txt','r')
   params = eval(fileParModule.read())

   fileParGeometry = open('modules_config/geometry_lime.txt','r')
   geo_params = eval(fileParGeometry.read())
   params.update(geo_params)

   print("full parameter set: ",params)
   
   fitter = ClusterMatcher(params)
   fitter.fitCluster(hits)

   # test the cluster matcher
   killer = Cluster(hits,1,None,None,'lime')
   killer.shapes['long_width'] = 300 # cluster shapes are not calculated when running standalone. Just set to a value to pass the threshold
   # this is to set the collection used in the reco
   killer.hits_fr_zs = hits
   targets = []
   for icl in range(7):
       if icl==cosm_cand:
           continue
       cluhits = np.load('debug_code/sclu_{i}.npy'.format(i=icl))
       cl = Cluster(cluhits,1,None,None,'lime')
       cl.hits_fr_zs = cluhits
       targets.append(cl)
   fitter.matchClusters(killer,targets)

   # check if a cluster has been killed
   for i,cl in enumerate(targets):
      print ("Cluster {icl} has min distance from a killer = {mdist:.1f} and number of robust (weak) matching pixels = {nr} ({nw}).".format(icl=i,mdist=cl.minDistKiller,nr=cl.nMatchKiller,nw=cl.nMatchKillerWeak))

