#!/usr/bin/env python

import numpy as np
import ROOT,math,os,sys,time
import pickle

from scipy.ndimage import gaussian_filter, median_filter
from skimage import img_as_float
from skimage.morphology import reconstruction
from skimage import measure

from morphsnakes import(morphological_chan_vese,
                        morphological_geodesic_active_contour,
                        inverse_gaussian_gradient,
                        checkerboard_level_set)

from clusterTools import Cluster
from cameraChannel import cameraTools
from iDBSCAN import iDBSCAN

import debug_code.tools_lib as tl

class SnakesFactory:
    def __init__(self,img,img_fr,img_fr_zs,img_ori,vignette,name,options,geometry):
        self.name = name
        self.options = options
        self.rebin = options.rebin
        self.geometry = geometry
        ct = cameraTools(geometry)
        self.image = img
        self.img_ori = img_ori
        self.imagelog = np.zeros((self.image.shape[0],self.image.shape[1]))
        for (x,y),value in np.ndenumerate(self.image):
            if value > 3.0/math.sqrt(self.rebin): # tresholding needed for tracking
                self.imagelog[x,y] = math.log(value)
        self.image_fr    = img_fr
        self.image_fr_zs = img_fr_zs
        self.vignette = vignette
        self.contours = []
        
    def store_evolution_in(self,lst):
        """Returns a callback function to store the evolution of the level sets in
        the given list.
        """
        def _store(x):
            lst.append(np.copy(x))
        return _store
    
    def getContours(self,iterations,threshold=0.69):
        
        # Morphological GAC
        image = img_as_float(self.image)
        gimage = inverse_gaussian_gradient(image)

        # Initial level set
        init_ls = np.zeros(image.shape, dtype=np.int8)
        init_ls[10:-10, 10:-10] = 1
        # List with intermediate results for plotting the evolution
        evolution = []
        callback = self.store_evolution_in(evolution)
        ls = morphological_geodesic_active_contour(gimage, iterations, init_ls,
                                                   smoothing=1, balloon=-1,
                                                   threshold=threshold,
                                                   iter_callback=callback)
        # before returning the snakes, put them in the event
        self.contours = ls
        return ls

    def getClusters(self,plot=False):

        from sklearn.cluster import DBSCAN
        from iDBSCAN import iDBSCAN
        from sklearn import metrics
        from scipy.spatial import distance
        from scipy.stats import pearsonr
        from random import random

        outname = self.options.plotDir
        if outname and not os.path.exists(outname):
            os.system("mkdir -p "+outname)
            os.system("cp utils/index.php "+outname)
        
        #   Plot parameters  #
        
        vmin=99
        vmax=125
        
        #   IDBSCAN parameters  #
        
        tip = self.options.tip
        
        scale              = 1
        iterative          = self.options.iterative                         # number of iterations for the IDBSC
        
        vector_eps         = self.options.vector_eps
        vector_min_samples = self.options.vector_min_samples

        vector_eps         = list(np.array(vector_eps, dtype=float)*scale)
        vector_min_samples = list(np.array(vector_min_samples, dtype=float)*scale)
        cuts               = self.options.cuts
        nb_it              = 3
        
        #-----Pre-Processing----------------#
        rescale=int(self.geometry.npixx/self.rebin)
        rebin_image     = tl.rebin(self.img_ori, (rescale, rescale))
        rebin_vignette  = tl.rebin(self.vignette,(rescale, rescale)) if self.options.vignetteCorr else np.ones(rescale, rescale)
        
        edges = median_filter(self.image, size=6)
        edcopy = edges.copy()
        edcopyTight = tl.noisereductor(edcopy,rescale,self.options.min_neighbors_average)

        # make the clustering with DBSCAN algo
        # this kills all macrobins with N photons < 1
        points = np.array(np.nonzero(np.round(edcopyTight))).astype(int).T
        lp = points.shape[0]

        if tip=='3D':
            Xl = [(ix,iy) for ix,iy in points]          # Aux variable to simulate the Z-dimension
            X1 = np.array(Xl).copy()                    # variable to keep the 2D coordinates
            for ix,iy in points:                        # Looping over the non-empty coordinates
                cut_vignette = min(rebin_vignette[ix,iy],2) # don't reduce the seeding threshold more than 1/2
                nreplicas = max(int(self.image[ix,iy]/cut_vignette)-1,0) # divide by the vignette correction the pixel to equivalently increase the "seeding threshold" of the cluster
                for count in range(nreplicas):                                # Looping over the number of 'photons' in that coordinate
                    Xl.append((ix,iy))                              # add a coordinate repeatedly 
            X = np.array(Xl)                                        # Convert the list to an array
        else:
            X = points.copy()
            X1 = X       
        
        if self.options.debug_mode == 0:
            self.options.flag_plot_noise = 0

        # returned collections
        clusters = []
        superclusters = []

        # clustering will crash if the vector of pixels is empty (it may happen after the zero-suppression + noise filtering)
        if len(X)==0:
            return clusters,superclusters

        t0 = time.perf_counter()
        # - - - - - - - - - - - - - -
        db = iDBSCAN(iterative = iterative, vector_eps = vector_eps, vector_min_samples = vector_min_samples, cuts = cuts, flag_plot_noise = self.options.flag_plot_noise).fit(X)
        t1 = time.perf_counter()
        if self.options.debug_mode: print(f"basic clustering in {t1 - t0:0.4f} seconds")
        
        if self.options.debug_mode == 1 and self.options.flag_plot_noise == 1:         
            for ext in ['png','pdf']:
                plt.savefig('{pdir}/{name}_{esp}.{ext}'.format(pdir=outname,name=self.name,esp='0th',ext=ext), bbox_inches='tight', pad_inches=0)
            plt.gcf().clear()
            plt.close('all')
        
        # Returning to '2' dimensions
        if tip == '3D':
            db.labels_              = db.labels_[range(0,lp)]               # Returning theses variables to the length
            db.tag_                 = db.tag_[range(0,lp)]                  # of the 'real' edges, to exclude the fake repetitions.
        # - - - - - - - - - - - - - -
        
        labels = db.labels_
        
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

        ##################### plot
        # the following is to preserve the square aspect ratio with all the camera pixels
        # plt.axes().set_aspect('equal','box')
        # plt.ylim(0,2040)
        # plt.xlim(0,2040)

        # Black removed and is used for noise instead.
        unique_labels = set(labels)

        colors = [(random(),random(),random(),1.0) for each in range(len(unique_labels))]

        # colors = [plt.cm.Spectral(each)
        #           for each in np.linspace(0, 1, len(unique_labels))]
        #canv = ROOT.TCanvas('c1','',600,600)
        if plot:
            #fig_edges = plt.figure(figsize=(10, 10))
            #plt.imshow(self.image.T, cmap='gray', vmin=0, vmax=1, origin='lower' ) 
            #plt.savefig('{pdir}/{name}_edges.png'.format(pdir=outname,name=self.name))
            fig = plt.figure(figsize=(10, 10))
            plt.imshow(self.image,cmap='viridis', vmin=1, vmax=25, interpolation=None, origin='lower' ) 
            #plt.savefig('{pdir}/{name}_edges.png'.format(pdir=outname,name=self.name))
            
        for k, col in zip(unique_labels, colors):
            if k == -1:
                col = [0, 0, 0, 1]
                break # noise: the unclustered

            class_member_mask = (labels == k)
         
            #xy = X[class_member_mask & core_samples_mask]
            xy = X1[class_member_mask]
            
            x = xy[:, 0]; y = xy[:, 1]
                            
            # only add the cores to the clusters saved in the event
            if k>-1 and len(x)>1:
                cl = Cluster(xy,self.rebin,self.image_fr,self.image_fr_zs,self.options.geometry,debug=False)
                cl.iteration = db.tag_[labels == k][0]
                cl.nclu = k
                
                #corr, p_value = pearsonr(x, y)
                cl.pearson = 999#p_value
                
                clusters.append(cl)
                if plot:
                    xri,yri = tl.getContours(y,x)
                    cline = {1:'r',2:'b',3:'y'}
                    plt.plot(xri,yri,'-{lcol}'.format(lcol=cline[cl.iteration]),linewidth=0.5)
                # if plot: cl.plotAxes(plot=plt,num_steps=100)
                # cl.calcProfiles(plot=None)
                # for dir in ['long','lat']:
                #     prof = cl.getProfile(dir)
                #     if prof and cl.widths[dir]>10: # plot the profiles only of sufficiently long snakes
                #         prof.Draw()
                #         for ext in ['png','pdf']:
                #             canv.SaveAs('{pdir}/{name}_snake{iclu}_{dir}profile.{ext}'.format(pdir=outname,name=self.name,iclu=k,dir=dir,ext=ext))

        t2 = time.perf_counter()
        if self.options.debug_mode: print(f"label basic clusters in {t2 - t1:0.4f} seconds")

        ## SUPERCLUSTERING
        from supercluster import SuperClusterAlgorithm
        superclusterContours = []
        scAlgo = SuperClusterAlgorithm(self.options,shape=rescale)
        u,indices = np.unique(db.labels_,return_index = True)
        allclusters_it1 = [X1[db.labels_ == i] for i in u[list(np.where(db.tag_[indices] == 1)[0])].tolist()]
        allclusters_it2 = [X1[db.labels_ == i] for i in u[list(np.where(db.tag_[indices] == 2)[0])].tolist()]
        allclusters_it12 = allclusters_it1 + allclusters_it2
        t3 = time.perf_counter()
        if self.options.debug_mode: print(f"supercl prep in {t3 - t2:0.4f} seconds")
        # note: passing the edges, not the filtered ones for deeper information
        superclusters,superclusterContours = scAlgo.findSuperClusters(allclusters_it12,edges,self.image_fr,self.image_fr_zs,0)

        t4 = time.perf_counter()
        if self.options.debug_mode: print(f"supercl in {t4 - t3:0.4f} seconds")
                
        if plot:
            for ext in ['png','pdf']:
                plt.savefig('{pdir}/{name}.{ext}'.format(pdir=outname,name=self.name,ext=ext), bbox_inches='tight', pad_inches=0)
            plt.gcf().clear()
            plt.close('all')
                        
        ## DEBUG MODE
        if self.options.debug_mode == 1:
            print('[DEBUG-MODE ON]')
            print('[%s Method]' % (self.options.tip))

            if self.options.flag_full_image or self.options.flag_rebin_image or self.options.flag_edges_image or self.options.flag_first_it or self.options.flag_second_it or self.options.flag_third_it or self.options.flag_all_it or self.options.flag_supercluster :
                import matplotlib.pyplot as plt

            if self.options.flag_full_image == 1:
                fig = plt.figure(figsize=(self.options.figsizeX, self.options.figsizeY))
                plt.imshow(np.flipud(self.image_fr),cmap=self.options.cmapcolor, vmin=1, vmax=25,origin='upper' )
                plt.title("Original Image")
                for ext in ['png','pdf']:
                    plt.savefig('{pdir}/{name}_{esp}.{ext}'.format(pdir=outname,name=self.name,esp='oriIma',ext=ext), bbox_inches='tight', pad_inches=0)
                with open('{pdir}/{name}_{esp}.pkl'.format(pdir=outname,name=self.name,esp='oriIma',ext=ext), "wb") as fp:
                    pickle.dump(fig, fp, protocol=4)
                plt.gcf().clear()
                plt.close('all')
                
            if self.options.flag_rebin_image == 1:
                fig = plt.figure(figsize=(self.options.figsizeX, self.options.figsizeY))
                plt.imshow(rebin_image,cmap=self.options.cmapcolor, vmin=vmin, vmax=vmax, origin='lower' )
                plt.title("Rebin Image")
                for ext in ['png','pdf']:
                    plt.savefig('{pdir}/{name}_{esp}.{ext}'.format(pdir=outname,name=self.name,esp='rebinIma',ext=ext), bbox_inches='tight', pad_inches=0)
                plt.gcf().clear()
                plt.close('all')
                
            if self.options.flag_edges_image == 1:
                fig = plt.figure(figsize=(self.options.figsizeX, self.options.figsizeY))
                plt.imshow(edcopyTight, cmap=self.options.cmapcolor, vmin=0, vmax=1, origin='lower' )
                plt.title('Edges after Filtering')
                for ext in ['png','pdf']:
                    plt.savefig('{pdir}/{name}_{esp}.{ext}'.format(pdir=outname,name=self.name,esp='edgesIma',ext=ext), bbox_inches='tight', pad_inches=0)
                plt.gcf().clear()
                plt.close('all')
                
            if self.options.flag_stats == 1:
                print('[Statistics]')
                n_clusters_ = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
                print("Total number of Clusters: %d" % (n_clusters_))
                u,indices = np.unique(db.labels_,return_index = True)
                print("Clusters found in iteration 1: %d" % (sum(db.tag_[indices] == 1)))
                print("Clusters found in iteration 2: %d" % (sum(db.tag_[indices] == 2)))
                print("Clusters found in iteration 3: %d" % (sum(db.tag_[indices] == 3)))
                print("SuperClusters found: %d" % len(superclusters))
                
            if self.options.flag_first_it == 1:
                print('[Plotting 1st iteration]')
                u,indices = np.unique(db.labels_,return_index = True)
                clu = [X1[db.labels_ == i] for i in u[list(np.where(db.tag_[indices] == 1)[0])].tolist()]

                fig = plt.figure(figsize=(self.options.figsizeX, self.options.figsizeY))
                plt.imshow(rebin_image,cmap=self.options.cmapcolor,vmin=vmin, vmax=vmax,origin='lower' )
                plt.title("Clusters found in iteration 1")

                for j in range(0,np.shape(clu)[0]):

                    ybox = clu[j][:,0]
                    xbox = clu[j][:,1]

                    if (len(ybox) > 0) and (len(xbox) > 0):
                        contours = tl.findedges(ybox,xbox,self.geometry.npixx,self.rebin)
                        for n, contour in enumerate(contours):
                            plt.plot(contour[:, 1],contour[:, 0], '-r',linewidth=2.5)

                for ext in ['png','pdf']:
                    plt.savefig('{pdir}/{name}_{esp}_{tip}.{ext}'.format(pdir=outname, name=self.name, esp='1st', ext=ext, tip=self.options.tip), bbox_inches='tight', pad_inches=0)
                with open('{pdir}/{name}_{esp}_{tip}.pkl'.format(pdir=outname,name=self.name,esp='1st',ext=ext,tip=self.options.tip), "wb") as fp:
                    pickle.dump(fig, fp, protocol=4)

                plt.gcf().clear()
                plt.close('all')
                
            if self.options.flag_second_it == 1:
                print('[Plotting 2nd iteration]')
                u,indices = np.unique(db.labels_,return_index = True)
                clu = [X1[db.labels_ == i] for i in u[list(np.where(db.tag_[indices] == 2)[0])].tolist()]

                fig = plt.figure(figsize=(self.options.figsizeX, self.options.figsizeY))
                plt.imshow(rebin_image,cmap=self.options.cmapcolor, vmin=vmin, vmax=vmax,origin='lower' )
                plt.title("Clusters found in iteration 2")

                for j in range(0,np.shape(clu)[0]):

                    ybox = clu[j][:,0]
                    xbox = clu[j][:,1]

                    if (len(ybox) > 0) and (len(xbox) > 0):
                        contours = tl.findedges(ybox,xbox,self.geometry.npixx,self.rebin)
                        for n, contour in enumerate(contours):
                            plt.plot(contour[:, 1],contour[:, 0], '-b',linewidth=2.5)

                for ext in ['png','pdf']:
                    plt.savefig('{pdir}/{name}_{esp}_{tip}.{ext}'.format(pdir=outname, name=self.name, esp='2nd', ext=ext, tip=self.options.tip), bbox_inches='tight', pad_inches=0)
                with open('{pdir}/{name}_{esp}_{tip}.pkl'.format(pdir=outname,name=self.name,esp='2nd',ext=ext,tip=self.options.tip), "wb") as fp:
                    pickle.dump(fig, fp, protocol=4)

                plt.gcf().clear()
                plt.close('all')
                    
                    
            if self.options.flag_third_it == 1:
                print('[Plotting 3rd iteration]')
                u,indices = np.unique(db.labels_,return_index = True)
                clu = [X1[db.labels_ == i] for i in u[list(np.where(db.tag_[indices] == 3)[0])].tolist()]

                fig = plt.figure(figsize=(self.options.figsizeX, self.options.figsizeY))
                plt.imshow(rebin_image,cmap=self.options.cmapcolor, vmin=vmin, vmax=vmax,origin='lower' )
                plt.title("Clusters found in iteration 3")

                for j in range(0,np.shape(clu)[0]):

                    ybox = clu[j][:,0]
                    xbox = clu[j][:,1]

                    if (len(ybox) > 0) and (len(xbox) > 0):
                        contours = tl.findedges(ybox,xbox,self.geometry.npixx,self.rebin)
                        for n, contour in enumerate(contours):
                            plt.plot(contour[:, 1],contour[:, 0], '-y',linewidth=2.5)

                for ext in ['png','pdf']:
                    plt.savefig('{pdir}/{name}_{esp}_{tip}.{ext}'.format(pdir=outname, name=self.name, esp='3rd', ext=ext, tip=self.options.tip), bbox_inches='tight', pad_inches=0)
                plt.gcf().clear()
                plt.close('all')
                
            if self.options.flag_all_it == 1:
                print('[Plotting ALL iteration]')
                u,indices = np.unique(db.labels_,return_index = True)
                clu = [X1[db.labels_ == i] for i in u[list(np.where(db.tag_[indices] == 1)[0])].tolist()]

                fig = plt.figure(figsize=(self.options.figsizeX, self.options.figsizeY))
                plt.imshow(rebin_image,cmap=self.options.cmapcolor, vmin=vmin, vmax=vmax,origin='lower' )
                plt.title("Final Image")

                for j in range(0,np.shape(clu)[0]):

                    ybox = clu[j][:,0]
                    xbox = clu[j][:,1]

                    if (len(ybox) > 0) and (len(xbox) > 0):
                        contours = tl.findedges(ybox,xbox,self.geometry.npixx,self.rebin)
                        for n, contour in enumerate(contours):
                            line, = plt.plot(contour[:, 1],contour[:, 0], '-r',linewidth=2.5)
                        if j == 0:
                            line.set_label('1st Iteration')

                clu = [X1[db.labels_ == i] for i in u[list(np.where(db.tag_[indices] == 2)[0])].tolist()]

                for j in range(0,np.shape(clu)[0]):

                    ybox = clu[j][:,0]
                    xbox = clu[j][:,1]
                    
                    if (len(ybox) > 0) and (len(xbox) > 0):
                        contours = tl.findedges(ybox,xbox,self.geometry.npixx,self.rebin)
                        for n, contour in enumerate(contours):
                            line, = plt.plot(contour[:, 1],contour[:, 0], '-b',linewidth=2.5)
                        if j == 0:
                            line.set_label('2nd Iteration')

                clu = [X1[db.labels_ == i] for i in u[list(np.where(db.tag_[indices] == 3)[0])].tolist()]

                for j in range(0,np.shape(clu)[0]):

                    ybox = clu[j][:,0]
                    xbox = clu[j][:,1]

                    if (len(ybox) > 0) and (len(xbox) > 0):
                        contours = tl.findedges(ybox,xbox,self.geometry.npixx,self.rebin)
                        for n, contour in enumerate(contours):
                            line, = plt.plot(contour[:, 1],contour[:, 0], '-y',linewidth=2.5)
                        if j == 0:
                            line.set_label('3rd Iteration')
                plt.legend(loc='upper left')

                if len(superclusters):
                    supercluster_contour = plt.contour(superclusterContours, [0.5], colors='limegreen', linewidths=2)
                    supercluster_contour.collections[0].set_label('supercluster')
                
                for ext in ['png','pdf']:
                    plt.savefig('{pdir}/{name}_{esp}_{tip}.{ext}'.format(pdir=outname, name=self.name, esp='all', ext=ext, tip=self.options.tip), bbox_inches='tight', pad_inches=0)
                with open('{pdir}/{name}_{esp}_{tip}.pkl'.format(pdir=outname,name=self.name,esp='all',ext=ext,tip=self.options.tip), "wb") as fp:
                    pickle.dump(fig, fp, protocol=4)

                plt.gcf().clear()
                plt.close('all')


            #################### PLOT SUPERCLUSTER ONLY ###############################
            if self.options.flag_supercluster == 1:
                if len(superclusters):
                    fig = plt.figure(figsize=(self.options.figsizeX, self.options.figsizeY))
                    supercluster_contour = plt.contour(superclusterContours, [0.5], colors='limegreen', linewidths=2,alpha=0.5)
                    #supercluster_contour.collections[0].set_label('supercluster it 1+2')
                    plt.imshow(rebin_image,cmap=self.options.cmapcolor,vmin=vmin, vmax=vmax,origin='lower' )
                    plt.title("Superclusters found")
                
                for ext in ['png','pdf']:
                    plt.savefig('{pdir}/{name}_{esp}_{tip}.{ext}'.format(pdir=outname, name=self.name, esp='sc', ext=ext, tip=self.options.tip), bbox_inches='tight', pad_inches=0)
                with open('{pdir}/{name}_{esp}_{tip}.pkl'.format(pdir=outname,name=self.name,esp='sc',ext=ext,tip=self.options.tip), "wb") as fp:
                    pickle.dump(fig, fp, protocol=4)

                plt.gcf().clear()
                plt.close('all')
            #################### PLOT SUPERCLUSTER ONLY ###############################

                
            if self.options.nclu >= 0:
                print('[Plotting just the cluster %d]' % (self.options.nclu))

                fig = plt.figure(figsize=(self.options.figsizeX, self.options.figsizeY))
                plt.imshow(rebin_image,cmap=self.options.cmapcolor, vmin=vmin, vmax=vmax,origin='lower' )
                plt.title('Plotting just the cluster %d' % (self.options.nclu))
                
                cl_mask = (db.labels_ == self.options.nclu)
         
                xy = X1[cl_mask]
                xbox = xy[:, 1] 
                ybox = xy[:, 0]

                if (len(ybox) > 0) and (len(xbox) > 0):
                    contours = tl.findedges(ybox,xbox,self.geometry.npixx,self.rebin)
                    for n, contour in enumerate(contours):
                        line, = plt.plot(contour[:, 1],contour[:, 0], '-r',linewidth=2.5)
                for ext in ['png','pdf']:
                    plt.savefig('{pdir}/{name}_{tip}_{nclu}.{ext}'.format(pdir=outname, name=self.name, ext=ext, tip = self.options.tip, nclu = self.options.nclu), bbox_inches='tight', pad_inches=0)
                plt.gcf().clear()
                plt.close('all')

        return clusters,superclusters
        
    def getTracks(self,plot=True):
        from skimage.transform import (hough_line, hough_line_peaks)
        # Classic straight-line Hough transform
        image = self.imagelog
        h, theta, d = hough_line(image)
        print("tracks found")
        
        tracks = []
        thr = 0.8 * np.amax(h)
        #######################   IMPLEMENT HERE THE SAVING OF THE TRACKS ############
        # loop over prominent tracks
        itrk = 0
        for _, angle, dist in zip(*hough_line_peaks(h, theta, d,threshold=thr)):
            print("Track # ",itrk)
            #points_along_trk = np.zeros((self.image.shape[1],self.image.shape[0]))
            points_along_trk = []
            for x in range(self.image.shape[1]):
                y = min(self.image.shape[0],max(0,int((dist - x * np.cos(angle)) / np.sin(angle))))
                #points_along_trk[x,y] = self.image[y,x]
                #print "adding point: %d,%d,%f" % (x,y,self.image[y,x])
                # add a halo fo +/- 20 pixels to calculate the lateral profile
                for iy in range(int(y)-5,int(y)+5):
                    if iy<0 or iy>=self.image.shape[0]: continue
                    points_along_trk.append((x,iy,self.image[iy,x]))
            xy = np.array(points_along_trk)
            trk = Cluster(xy,self.rebin)
            tracks.append(trk)
            itrk += 1
        ###################################
            
        if plot:
            # Generating figure
            from matplotlib import cm
            fig, ax = plt.subplots(2, 1, figsize=(18, 6))
            #ax = axes.ravel()

            ax[0].imshow(image, cmap=cm.gray)
            ax[0].set_title('Camera image')
            #ax[0].set_axis_off()            

            ax[1].imshow(image, cmap=cm.gray)
            for _, angle, dist in zip(*hough_line_peaks(h, theta, d,threshold=thr)):
                y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
                y1 = (dist - image.shape[1] * np.cos(angle)) / np.sin(angle)
                ax[1].plot((0, image.shape[1]), (y0, y1), '-r')
            ax[1].set_xlim((0, image.shape[1]))
            ax[1].set_ylim((image.shape[0], 0))
            #ax[1].set_axis_off()
            ax[1].set_title('Fitted tracks')

            plt.tight_layout()
            #plt.show()
            outname = self.options.plotDir
            if outname and not os.path.exists(outname):
                os.system("mkdir -p "+outname)
                os.system("cp ~/cernbox/www/Cygnus/index.php "+outname)
            for ext in ['pdf']:
                plt.savefig('{pdir}/{name}.{ext}'.format(pdir=outname,name=self.name,ext=ext))
            plt.gcf().clear()

        return tracks
        
    def plotClusterFullResolution(self,clusters):
        outname = self.options.plotDir
        for k,cl in enumerate(clusters):
            cl.plotFullResolution('{pdir}/{name}_cluster{iclu}'.format(pdir=outname,name=self.name,iclu=k))

    def calcProfiles(self,clusters,plot=False):
        for k,cl in enumerate(clusters):
            profName = '{name}_cluster{iclu}'.format(name=self.name,iclu=k)
            cl.calcProfiles(name=profName,plot=plot)
                             
    def plotProfiles(self,clusters):
        print ("plot profiles...")
        outname = self.options.plotDir
        canv = ROOT.TCanvas('c1','',1200,600)
        for k,cl in enumerate(clusters):
            for dir in ['long','lat']:
                profName = '{name}_cluster{iclu}_{dir}'.format(name=self.name,iclu=k,dir=dir)
                prof = cl.getProfile(dir)
                if prof and cl.widths[dir]>0.2: # plot the profiles only of sufficiently long snakes (>200 um)
                    prof.Draw("pe1")
                    for ext in ['pdf']:
                        canv.SaveAs('{pdir}/{name}profile.{ext}'.format(pdir=outname,name=profName,ext=ext))
        
    def plotContours(self,contours):

        image = img_as_float(self.image)
        #fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        #ax = axes.flatten()
        fig, ax = plt.subplots()
        
        ax.imshow(image, cmap="gray")
        ax.set_axis_off()
        ax.contour(contours, [0.5], colors='r')
        #ax.set_title("Morphological GAC segmentation", fontsize=12)
        (run,event) = self.name.split('_')
        ax.set_title('Run={run}, Event={event}'.format(run=run,event=event), fontsize=12)
        
        # ax[1].imshow(contours, cmap="gray")
        # ax[1].set_axis_off()
        # contour = ax[1].contour(evolution[0], [0.5], colors='g')
        # contour.collections[0].set_label("Iteration 0")
        # contour = ax[1].contour(evolution[50], [0.5], colors='y')
        # contour.collections[0].set_label("Iteration 50")
        # contour = ax[1].contour(evolution[-1], [0.5], colors='r')
        # contour.collections[0].set_label("Iteration 100")
        # ax[1].legend(loc="upper right")
        # title = "Morphological GAC evolution"
        # ax[1].set_title(title, fontsize=12)
        
        fig.tight_layout()
        #plt.show()
        for ext in ['pdf']:
            plt.savefig('{name}.{ext}'.format(name=self.name,ext=ext))



class SnakesProducer:
    def __init__(self,sources,params,options,geometry):
        self.picture     = sources['picture']     if 'picture' in sources else None
        self.pictureHD   = sources['pictureHD']   if 'pictureHD' in sources else None
        self.picturezsHD = sources['picturezsHD'] if 'picturezsHD' in sources else None
        self.pictureOri  = sources['pictureOri']  if 'pictureOri' in sources else None
        self.vignette    = sources['vignette']    if 'vignette' in sources else None
        self.name        = sources['name']        if 'name' in sources else None
        self.algo        = sources['algo']        if 'algo' in sources else 'DBSCAN'
        
        self.snakeQualityLevel = params['snake_qual']   if 'snake_qual' in params else 3
        self.plot2D            = params['plot2D']       if 'plot2D' in params else False
        self.plotpy            = params['plotpy']       if 'plotpy' in params else False
        self.plotprofiles      = params['plotprofiles'] if 'plotprofiles' in params else False

        self.options = options
        self.geometry = geometry
        geometryPSet   = open('modules_config/geometry_{det}.txt'.format(det=options.geometry),'r')
        geometryParams = eval(geometryPSet.read())

        self.run_cosmic_killer = self.options.cosmic_killer
        if self.run_cosmic_killer:
            from clusterMatcher import ClusterMatcher
            # cosmic killer parameters
            cosmicKillerPars = open('modules_config/clusterMatcher.txt','r')
            killer_params = eval(cosmicKillerPars.read())
            killer_params.update(geometryParams)
            self.cosmic_killer = ClusterMatcher(killer_params)

        
    def run(self):
        ret = []
        if any([x==None for x in (self.picture.any(),self.pictureHD.any(),self.picturezsHD.any(),self.name)]):
            return ret

        t0 = time.perf_counter()
        
        # Cluster reconstruction on 2D picture
        snfac = SnakesFactory(self.picture,self.pictureHD,self.picturezsHD,self.pictureOri,self.vignette,self.name,self.options,self.geometry)

        # this plotting is only the pyplot representation.
        # Doesn't work on MacOS with multithreading for some reason... 
        if self.algo=='DBSCAN':
            clusters,snakes = snfac.getClusters(plot=self.plotpy)
            
        elif self.algo=='HOUGH':
            clusters = []
            snakes = snfac.getTracks(plot=self.plotpy)            
        t1 = time.perf_counter()
        if self.options.debug_mode: print(f"FULL RECO in {t1 - t0:0.4f} seconds")
            
        # print "Get light profiles..."
        snfac.calcProfiles(snakes,plot=self.plotpy)
        snfac.calcProfiles(clusters,plot=False)

        # run the cosmic killer: it makes sense only on superclusters
        if self.run_cosmic_killer:
            for ik,killerCand in enumerate(snakes):
                targets = [snakes[it] for it in range(len(snakes)) if it!=ik]
                self.cosmic_killer.matchClusters(killerCand,targets)

        # snfac.calcProfiles(snakes) # this is for BTF
        
        # sort snakes by light integral
        snakes = sorted(snakes, key = lambda x: x.integral(), reverse=True)
        # and reject discharges (round)
        #snakes = [x for x in snakes if x.qualityLevel()>=self.snakeQualityLevel]
        
        # plotting
        if self.plot2D:       snfac.plotClusterFullResolution(snakes)
        if self.plotprofiles: snfac.plotProfiles(snakes)

        return clusters,snakes
