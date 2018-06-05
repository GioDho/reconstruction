#!/usr/bin/env python

import os,math
import numpy as np
import matplotlib.pyplot as plt
import ROOT
ROOT.gROOT.SetBatch(True)


from snakes import SnakesFactory

class analysis:

    def __init__(self,rfile,options):
        self.rebin = options.rebin        
        self.rfile = rfile
        self.options = options
        self.pedfile_name = '{base}_ped_rebin{rb}.root'.format(base=os.path.splitext(self.rfile)[0],rb=self.rebin)
        if not options.calcPedestals and not os.path.exists(self.pedfile_name):
            print "WARNING: pedestal file ",self.pedfile_name, " not existing. First calculate them..."
            self.calcPedestal()
        print "Pulling pedestals..."
        pedrf = ROOT.TFile.Open(self.pedfile_name)
        self.pedmap = pedrf.Get('pedmap').Clone()
        self.pedmap.SetDirectory(None)
        self.pedmean = pedrf.Get('pedmean').GetMean()
        self.pedrms = pedrf.Get('pedrms').GetMean()
        pedrf.Close()
        
    def zs(self,th2):
        nx = th2.GetNbinsX(); ny = th2.GetNbinsY();
        th2_zs = ROOT.TH2D(th2.GetName()+'_zs',th2.GetName()+'_zs',nx,0,nx,ny,0,ny)
        th2_zs.SetDirectory(None)
        for ix in xrange(1,nx+1):
            for iy in xrange(1,ny+1):
                if not self.isGoodChannel(ix,iy): continue
                ped = self.pedmap.GetBinContent(ix,iy)
                noise = self.pedmap.GetBinError(ix,iy)
                z = max(th2.GetBinContent(ix,iy)-ped,0)
                if z>5*noise: th2_zs.SetBinContent(ix,iy,z)
                #print "x,y,z=",ix," ",iy," ",z,"   3*noise = ",3*noise
        th2_zs.GetZaxis().SetRangeUser(0,1)
        return th2_zs

    def calcPedestal(self,maxImages=-1):
        nx=ny=2048
        nx=int(nx/self.rebin); ny=int(ny/self.rebin); 
        pedfile = ROOT.TFile.Open(self.pedfile_name,'recreate')
        pedmap = ROOT.TProfile2D('pedmap','pedmap',nx,0,nx,ny,0,ny,'s')
        tf = ROOT.TFile.Open(self.rfile)
        for i,e in enumerate(tf.GetListOfKeys()):
            if maxImages>-1 and i==maxImages: break
            name=e.GetName()
            obj=e.ReadObj()
            if not obj.InheritsFrom('TH2'): continue
            print "Processing histogram: ",name
            obj.RebinX(self.rebin); obj.RebinY(self.rebin); 
            for ix in xrange(nx):
                for iy in xrange(ny):
                    pedmap.Fill(ix,iy,obj.GetBinContent(ix+1,iy+1)/float(math.pow(self.rebin,2)))

        tf.Close()
        pedfile.cd()
        pedmap.Write()
        pedmean = ROOT.TH1D('pedmean','pedestal mean',500,97,103)
        pedrms = ROOT.TH1D('pedrms','pedestal RMS',500,0,5)
        for ix in xrange(nx):
            for iy in xrange(ny):
               pedmean.Fill(pedmap.GetBinContent(ix,iy)) 
               pedrms.Fill(pedmap.GetBinError(ix,iy)) 
        pedmean.Write()
        pedrms.Write()
        pedfile.Close()

    def isGoodChannel(self,ix,iy):
        pedval = self.pedmap.GetBinContent(ix,iy)
        pedrms = self.pedmap.GetBinError(ix,iy)
        if pedval > 110: return False
        if pedrms < 0.2: return False
        if pedrms > 5: return False
        return True

    def reconstruct(self):
        ROOT.gStyle.SetOptStat(0)
        ROOT.gStyle.SetPalette(ROOT.kRainBow)
        tf = ROOT.TFile.Open(self.rfile)
        c1 = ROOT.TCanvas('c1','',600,600)
        # loop over events (pictures)
        for ie,e in enumerate(tf.GetListOfKeys()) :
            if ie==options.maxEntries: break
            name=e.GetName()
            obj=e.ReadObj()
            if not obj.InheritsFrom('TH2'): continue
            print "Processing histogram: ",name
            obj.RebinX(self.rebin); obj.RebinY(self.rebin)
            obj.Scale(1./float(math.pow(self.rebin,2)))
            h2zs = self.zs(obj)

            print "Analyzing its contours..."
            snfac = SnakesFactory(h2zs,name)
            snakes = snfac.getContours(iterations=100)
            snfac.plotContours(snakes,fill=True)
            snfac.filledSnakes(snakes)
            
if __name__ == '__main__':

    from optparse import OptionParser
    parser = OptionParser(usage='%prog h5file1,...,h5fileN [opts] ')
    parser.add_option('-r', '--rebin', dest='rebin', default=10, type='int', help='Rebin factor (same in x and y)')
    parser.add_option('-p', '--pedestal', dest='calcPedestals', default=False, action='store_true', help='First calculate the pedestals')
    parser.add_option(      '--max-entries', dest='maxEntries', default=-1,type='float', help='Process only the first n entries')
    (options, args) = parser.parse_args()

    inputf = args[0]
    ana = analysis(inputf,options)
    if options.calcPedestals:
        ana.calcPedestal()
    ana.reconstruct()