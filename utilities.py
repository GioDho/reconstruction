#!/usr/bin/env python
import subprocess

import os,sys,optparse
import numpy as np
from root_numpy import hist2array
import ROOT
import swiftlib as sw
from cameraChannel import cameraTools, cameraGeometry

class utils:
    def __init__(self):
        pass

    def dynamicProfileBins(self,hits,coord='x',relError=0.1):
        minPixels = max(1,1/relError/relError)
        index = 0 if coord=='x' else 1
        xmin=min([h[index] for h in hits])
        xmax=max([h[index] for h in hits])
        x=int(xmin)
        xedges=[x]
        integral=0
        while x<xmax:
            if integral<minPixels:
                integral += sum([int(h[index])==int(x) for h in hits])
            else:
                xedges.append(x)
                integral=0
            x+=1
        xedges.append(int(xmax))
        return xedges

    def rotate_around_point(self, hit, dir, pivot, inverse=False):
        x,y = hit[:-1]
        ox, oy = pivot
        cos,sin = dir
        if inverse: cos = -1*cos
        qx = ox + cos * (x - ox) + sin * (y - oy)
        qy = oy - sin * (x - ox) + cos * (y - oy)
        return qx, qy

    def gen_rand_limit(self, x1, x2, y1, y2, maxx=2048, maxy=2048):
        import random
        # generate x, y O(1)
        # --x
        left = random.randrange(0, x1)
        right = random.randrange(x2+1, maxx)
        withinx = random.randrange(x1, x2+1)
        # adjust probability of a point outside the box columns
        # a point outside has probability (1/(maxx-w)) v.s. a point inside has 1/w
        # the same is true for rows. adjupx/y adjust for this probability 
        w = abs(x2-x1)
        h = abs(y2-y1)
        adjpx = ((maxx - w)/w/2)
        x = random.choice([left, right] * adjpx + [withinx])
        # --y
        top = random.randrange(0, y1)
        bottom = random.randrange(y2+1, maxy)
        withiny = random.randrange(y1, y2+1)
        if x == left or x == right:
            adjpy = ((maxy- h)/h/2)
            y = random.choice([top, bottom] * adjpy + [withiny])
        else:
            y = random.choice([top, bottom])
        return x, y 

    def get_git_revision_hash(self):
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'])

    def calcVignettingMap(self,run,pedfile,outfile,maxImages=1000,N=2304,rebin=12,det='lime',daq='midas'):

        ################ GEOMETRY ###
        geometryPSet   = open('modules_config/geometry_{det}.txt'.format(det=det),'r')
        geometryParams = eval(geometryPSet.read())
        cg = cameraGeometry(geometryParams)
        ctools = cameraTools(cg)
        #############################
        
        # pedestal map, full reso
        pedrf_fr = ROOT.TFile.Open(pedfile)
        pedmap_fr = pedrf_fr.Get('pedmap').Clone()
        pedmap_fr.SetDirectory(0)
        pedarr_fr = hist2array(pedmap_fr).T
        noisearr_fr = ctools.noisearray(pedmap_fr).T
        pedrf_fr.Close()
     
        
        outname_base = os.path.basename(outfile).split('.')[0]
        tf_out = ROOT.TFile.Open(outname_base+'.root','recreate')
     
        nx=int(N/rebin); ny=int(N/rebin);
        normmap = ROOT.TH2D('normmap','normmap',nx,0,N,nx,0,N)
        
        mapsum = np.zeros((nx,nx))
     
        if sw.checkfiletmp(int(run)):
            infile = "/tmp/histograms_Run%05d.root" % int(run)
        else:
            print ('Downloading file: ' + sw.swift_root_file('Data', int(run)))
            infile = sw.swift_download_root_file(sw.swift_root_file('Data', int(run)),int(run))
            
        tf_in = sw.swift_read_root_file(infile)
     
        framesize = 216 if det=='lime' else 0
        
        # first calculate the mean 
        for i,e in enumerate(tf_in.GetListOfKeys()):
            iev = i if daq != 'midas'  else i/2 # when PMT is present
     
            if maxImages>-1 and i<len(tf_in.GetListOfKeys())-maxImages: continue
            name=e.GetName()
            obj=e.ReadObj()
            if not obj.InheritsFrom('TH2'): continue
            print("Calc pixel sums with event: ",name)
            arr = hist2array(obj)
            
            # Upper Threshold full image
            img_cimax = np.where(arr < 300, arr, 0)
            img_fr_sub = ctools.pedsub(img_cimax,pedarr_fr)
            img_fr_zs  = ctools.zsfullres(img_fr_sub,noisearr_fr,nsigma=1)
     
            # for lime, remove the borders of the sensor
            if det=='lime':
                #img_fr_zs[:framesize,:]=0
                #img_fr_zs[-framesize:,:]=0
                img_fr_zs[:,:framesize]=0
                img_fr_zs[:,-framesize:]=0
                
            img_rb_zs  = ctools.arrrebin(img_fr_zs,rebin)
            mapsum = np.add(mapsum,img_rb_zs)
     
        # calc the normalized map wrt the center area
        CA = 96
        central_square = mapsum[int((N-CA)/rebin/2):int((N+CA)/rebin/2),int((N-CA)/rebin/2):int((N+CA)/rebin/2)]
        norm = np.mean(central_square)
        print ("Now normalizing to the central area value = ",norm)
        mapnorm = mapsum / float(norm)
        if det=='lime':
            framesize_rb = int(framesize/rebin)
            #mapnorm[:framesize_rb,:]=1
            #mapnorm[-framesize_rb:,:]=1
            mapnorm[:,:framesize_rb]=1
            mapnorm[:,-framesize_rb:]=1
        
        # now save in a persistent ROOT object. Threshold to 1
        for ix in range(nx):
            for iy in range(nx):
                normmap.SetBinContent(ix+1,iy+1,min(mapnorm[ix,iy],1.));
        tf_in.Close()
     
        tf_out.cd()
        normmap.Write()
        tf_out.Close()
        print("Written the mean map with rebinning {rb}x{rb} into file {outf}.".format(rb=rebin,outf=outfile))



class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

if __name__ == "__main__":
    parser = optparse.OptionParser(usage='usage: %prog [opts] ', version='%prog 1.0')
    parser.add_option('', '--make'   , type='string'       , default='calcVignette' , help='run utilities.py (options = calcVignette)')
    (options, args) = parser.parse_args()

    if options.make == 'calcVignette':
        run = 3806
        pedfile = 'pedestals/pedmap_run3807_rebin1.root'
        ut = utils()
        ut.calcVignettingMap(run,"pedestals/pedmap_run3797_rebin1.root","vignette_run%05d.root" % run)

        
