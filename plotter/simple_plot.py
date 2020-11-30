import os, math, optparse, ROOT
from array import array
from root_numpy import hist2array
import numpy as np

ROOT.gStyle.SetOptStat(111111)
ROOT.gROOT.SetBatch(True)

# if "/functions_cc.so" not in ROOT.gSystem.GetLibraries(): 
#     ROOT.gROOT.ProcessLine(".L functions.cc+")

GEOMETRY = 'lime'
NX = {'lime':  2304,
      'lemon': 2048 }

fe_integral_rescale = 0.1
cosm_rate_calib = [1.13,0.02]     # central value, stat error

### INTERCALIBRATION CONSTANTS CALCULATED WITH COSMICS ###
# this comes from fitting the raw energy density in the CR. The uncertainty is negligible.

### inter-calib with cosmics after 2hrs
# cosm_energycalib = {'fe'   : 21.83/14.95,
#                     'cosm' : 21.83/19.81,
#                     'ambe' : 1 }
### inter-calib with cosmics soon after AmBe
cosm_energycalib = {'fe'   : 21.83/14.97,
                    'cosm' : 21.83/21.31,
                    'ambe' : 1 }
### no inter-calib
#cosm_energycalib = {'fe'   : 1,
#                    'cosm' : 1,
#                    'ambe' : 1 }
##########################################################

### ABSOLUTE ENERGY SCALE ###
fe_truth = 5.9 # keV
fe_integral_mpv = 1800. # number of photons most probable value
fe_calib_gaussigma = [1.07,0.014] # central value, stat error
fe_calib_gauspeak  = [fe_truth*1000./cosm_energycalib['fe']/fe_integral_mpv,0.013] # central value, stat error in eV/ph
##########################################################

# rescaling for cosmics dE/dx MPV to be 2.3 keV/cm
cosmics_cal =  0.29
fe_calib_gauspeak = [cosmics_cal * x for x in fe_calib_gauspeak]

pixw = 0.152 if GEOMETRY=='lime' else 0.125 # mm

def printTLatex(tl,color=ROOT.kBlack):
    tl.SetNDC()
    tl.SetTextFont(42)
    tl.SetTextAlign(21)
    tl.SetTextSize(0.03)
    tl.SetTextColor(color)
    tl.Draw()

def angleWrtHorizontal(xmin,xmax,ymin,ymax):
    x = xmax-xmin
    y = ymax-ymin
    length = math.hypot(x,y)
    ## figure is rotated by 90 degrees
    return math.asin(y/length)*180/3.14

def saturationFactor(density):
    return 1.5/1.8 * math.pow(5979 * math.log(20512./(20512.-density)),1.8)/(0.27*density-0.1)

def saturationFactorNLO(density):
    if density<=0: # protection for the formula below
        ret = 0.85 # seems to provide some continuity
    else:
        x = density/1.5
        ret = (0.11 + 1.22*x)/(130283. * (1-math.exp(-1*(math.pow(x,0.56757)/117038.))))/1.2
    ## rescale the absolute scale for the observed Fe peak
    return ret * fe_truth / fe_calib_gauspeak[0]

# this is for LIME
# attach to a dict to make it persistent
vignetteCorrMap = { 'lime' : np.zeros((int(NX['lime']/16),int(NX['lime']/16))) }

def vignettingCorrection(x,y,length,detector='lime'):
    ## since in these ntuples we have only xmean,ymean, not the slices,
    ## we don't apply to the cosmics, which are long, and typically cross the center.
    if length>5:
        return 1
    corr = 0
    if detector == 'lemon':
        corr = 1
    elif detector == 'lime':
        if not vignetteCorrMap[detector].any():
            print("get vignetting map...")
            tf = ROOT.TFile.Open('../data/vignette_run03806.root')
            hmap = tf.Get('normmap')
            vignetteCorrMap[detector] = hist2array(hmap)
        rebinx = NX[GEOMETRY]/vignetteCorrMap[detector].shape[0]
        rebiny = NX[GEOMETRY]/vignetteCorrMap[detector].shape[1]
        #print ("rebin = ",rebinx," ",rebiny)
        #print ("x = {x}, y={y}, bx={bx}, by={by}".format(x=x,y=y,bx=int(x/rebinx),by=int(y/rebiny)))
        corr = 1./(vignetteCorrMap[detector])[int(x/rebinx),int(y/rebiny)]
        #print ("x = {x}, y={y}, corr = {corr}".format(x=x,y=y,corr=corr))
    else:
        print ('WARNING! Detector ',detector,' not foreseen. Return correction 1')
        corr = 1
    return corr

def is60keVBkg(length,density):
    # rough linear decrease density vs length
    central = 14. - length/50.
    # take +/- 2 band around the central value
    return (central-2. < density < central+2.) and 120 < length < 250 

def spotsLowDensity(length,density):
    return length < 80 and 5<density<8

def cosmicSelection(length,gsigma):
    return length*pixw>70 and gsigma<1

def isPurpleBlob(length,density):
    # rough linear decrease density vs length
    central = 10 - length/50.
    # take +/- 2 band around the central value
    return (central-1.5 < density < central+1.5) and 90 < length < 250 

def doLegend(histos,labels,styles,corner="TR",textSize=0.035,legWidth=0.18,legBorder=False,nColumns=1):
    nentries = len(histos)
    (x1,y1,x2,y2) = (.85-legWidth, .7 - textSize*max(nentries-3,0), .90, .89)
    if corner == "TR":
        (x1,y1,x2,y2) = (.85-legWidth, .7 - textSize*max(nentries-3,0), .90, .89)
    elif corner == "TC":
        (x1,y1,x2,y2) = (.5, .7 - textSize*max(nentries-3,0), .5+legWidth, .89)
    elif corner == "TL":
        (x1,y1,x2,y2) = (.2, .7 - textSize*max(nentries-3,0), .2+legWidth, .89)
    elif corner == "BR":
        (x1,y1,x2,y2) = (.85-legWidth, .15 + textSize*max(nentries-3,0), .90, .25)
    elif corner == "BC":
        (x1,y1,x2,y2) = (.5, .15 + textSize*max(nentries-3,0), .5+legWidth, .25)
    elif corner == "BL":
        (x1,y1,x2,y2) = (.2, .23 + textSize*max(nentries-3,0), .33+legWidth, .35)
    leg = ROOT.TLegend(x1,y1,x2,y2)
    leg.SetNColumns(nColumns)
    leg.SetFillColor(0)
    leg.SetFillColorAlpha(0,0.6)  # should make the legend semitransparent (second number is 0 for fully transparent, 1 for full opaque)
    #leg.SetFillStyle(0) # transparent legend, so it will not cover plots (markers of legend entries will cover it unless one changes the histogram FillStyle, but this has other effects on color, so better not touching the FillStyle)
    leg.SetShadowColor(0)
    if not legBorder:
        leg.SetLineColor(0)
        leg.SetBorderSize(0)  # remove border  (otherwise it is drawn with a white line, visible if it overlaps with plots
    leg.SetTextFont(42)
    leg.SetTextSize(textSize)
    for (plot,label,style) in zip(histos,labels,styles): leg.AddEntry(plot,label,style)
    leg.Draw()
    ## assign it to a global variable so it's not deleted
    global legend_
    legend_ = leg
    return leg


def plotDensity():
    tf = ROOT.TFile('reco_run815.root')
    tree = tf.Get('Events')

    histos = []
    colors = [ROOT.kRed,ROOT.kBlue,ROOT.kOrange]
    #histo = ROOT.TH1F('density','',70,0,2500)
    histo = ROOT.TH1F('density','',70,0,20)
    for it in range(1,4):
        h = histo.Clone('h_iter{it}'.format(it=it))
        h.Sumw2()
        #tree.Draw('track_integral/track_nhits>>h_iter{it}'.format(it=it),'track_iteration=={it}'.format(it=it))
        #tree.Draw('track_integral/track_length>>h_iter{it}'.format(it=it),'track_iteration=={it}'.format(it=it))
        tree.Draw('track_length>>h_iter{it}'.format(it=it),'track_iteration=={it}'.format(it=it))
        h.Scale(1./h.Integral())
        h.SetFillColor(colors[it-1])
        h.SetLineColor(colors[it-1])
        h.SetFillStyle(3005)
        histos.append(h)


    # legend
    (x1,y1,x2,y2) = (0.7, .70, .9, .87)
    leg = ROOT.TLegend(x1,y1,x2,y2)
    leg.SetFillColor(0)
    leg.SetFillColorAlpha(0,0.6)
    leg.SetShadowColor(0)
    leg.SetLineColor(0)
    leg.SetBorderSize(0)
    leg.SetTextFont(42)
    leg.SetTextSize(0.035)
    
    c = ROOT.TCanvas('c','',600,600)
    for ih,h in enumerate(histos):
        h.Draw('hist' if ih==0 else 'hist same')
        h.GetYaxis().SetRangeUser(0,0.25)
        h.GetXaxis().SetTitle('length (mm)')
        leg.AddEntry(h,'iteration {it}'.format(it=ih+1),'f')

    leg.Draw()

    c.SaveAs('density.pdf')


def plotNClusters(iteration=1):

    f = ROOT.TFile.Open('../runs/fng_runs.root')
    nclu_h = ROOT.TH1F('nclu_h','',20,0,80)
    
    for ie,event in enumerate(f.Events):
        #if event.run<46 or event.run>47: continue
        #if event.run<48 or event.run>50: continue
        if event.run<70 or event.run>71: continue
        nclu_it1 = 0
        for icl in range(event.nTrack):
            if event.track_nhits[icl]<100: continue
            if int(event.track_iteration[icl])>2: continue
            if math.hypot(event.track_xmean[icl]-1024,event.track_ymean[icl]-1024)>1000: continue
            #print "cluster x = ", event.track_xmean[icl]
            #print "cluster y = ", event.track_ymean[icl]
            nclu_it1 += 1
        nclu_h.Fill(nclu_it1)

    c = ROOT.TCanvas('c1','',600,400)
    nclu_h.SetLineColor(ROOT.kRed+2)
    nclu_h.SetMarkerColor(ROOT.kBlack)
    nclu_h.SetMarkerSize(1.5)
    nclu_h.SetMarkerStyle(ROOT.kOpenCircle)
    nclu_h.SetLineWidth(2)
    nclu_h.GetXaxis().SetTitle('# clusters / 100 ms')
    nclu_h.GetYaxis().SetTitle('Events')
    nclu_h.Draw('pe')
    nclu_h.Draw('hist same')
    c.SaveAs("nclusters_iter1_run70_71.pdf")

def withinFC(xmean,ymean,ax=480,ay=600,shape=NX[GEOMETRY]):
    center = shape/2.
    x1 = xmean-center
    y1 = (ymean-center)*1.2
    return math.hypot(x1,y1)<ax

def withinFCFull(xmin,ymin,xmax,ymax,ax=480,ay=600,shape=NX[GEOMETRY]):
    center = shape/2.
    x1 = xmin-center
    y1 = (ymin-center)*1.2
    x2 = xmax-center
    y2 = (ymax-center)*1.2
    return math.hypot(x1,y1)<ax and math.hypot(x2,y2)<ax

def limeQuietRegion(xmean,ymean,framesize=256):
    return xmean>framesize and xmean<NX[GEOMETRY]-framesize and ymean>framesize and ymean<NX[GEOMETRY]-framesize

def slimnessCut(l,w,th=0.6):
    
        return (w/l)>th

def integralCut(i,minTh=1800,maxTh=3200):
    
        return (i>minTh) & (i<maxTh)
    
def varChoice(var):
    
    if var == 'integral':
        var1 = 'cl_integral'
        leg = 'Integral [ph]'
        histlimit = 6000
    elif var == 'length':
        var1 = 'cl_length*125E-3'
        leg = 'Length [mm]'
        histlimit = 32
    elif var == 'width':
        var1 = 'cl_width*125E-3'
        leg = 'Width [mm]'
        histlimit = 10
    elif var == 'size':
        var1 = 'cl_size'
        leg = 'Size [px]'
        histlimit = 1600
    elif var == 'slimness':
        var1 = 'cl_width/cl_length'
        leg = 'Slimness [w/l]'
        histlimit = 1.2
    else:
        exit()
    
    return var1, leg, histlimit


def fillSpectra(cluster='sc'):

    ret = {}
    data_dir = '/Users/emanuele/Work/data/cygnus/RECO/lime2020/'
    tf_ambe  = ROOT.TFile('{d}/ambe_lime.root'.format(d=data_dir))
    tf_cosmics = ROOT.TFile('{d}/cosmics_firstruns_lime.root'.format(d=data_dir))
    tf_fe55 = ROOT.TFile('{d}/fe_lime.root'.format(d=data_dir))

    tfiles = {'fe':tf_fe55,'ambe':tf_ambe,'cosm':tf_cosmics}

    entries = {'fe': tf_fe55.Events.GetEntries(), 'ambe': tf_ambe.Events.GetEntries(), 'cosm': tf_cosmics.Events.GetEntries()}
    
    ## Fe55 region histograms
    ret[('ambe','integral')] = ROOT.TH1F("integral",'',50,0,1e4)
    ret[('ambe','integralExt')] = ROOT.TH1F("integralExt",'',50,0,25e4)
    ret[('ambe','calintegral')] = ROOT.TH1F("calintegral",'',100,0,20)
    ret[('ambe','energy')] = ROOT.TH1F("energy",'',50,0,60)
    ret[('ambe','calintegralExt')] = ROOT.TH1F("calintegralExt",'',50,0,1000)
    ret[('ambe','energyExt')] = ROOT.TH1F("energyExt",'',50,0,200)
    #energyBins = array('f',list(range(11))+list(range(12,17,2))+list(range(19,32,3))+list(range(40,101,10))+list(range(150,201,50)))
    energyBins = array('f',list(range(11))+list(range(12,17,2))+list(range(19,32,3))+list(range(40,51,10))+list(range(75,149,25))+list(range(150,201,50)))
    ret[('ambe','energyFull')] = ROOT.TH1F("energyFull",'',len(energyBins)-1,energyBins)
    ret[('ambe','length')]   = ROOT.TH1F("length",'',50,0,NX[GEOMETRY]*pixw)
    ret[('ambe','width')]    = ROOT.TH1F("width",'',100,0,200*pixw)
    ret[('ambe','tgausssigma')]    = ROOT.TH1F("tgausssigma",'',100,0,40*pixw)
    ret[('ambe','nhits')]    = ROOT.TH1F("nhits",'',50,0,10000)
    ret[('ambe','slimness')] = ROOT.TH1F("slimness",'',50,0,1)
    ret[('ambe','density')]  = ROOT.TH1F("density",'',50,10,50)
    ret[('ambe','caldensity')]  = ROOT.TH1F("caldensity",'',60,0,40)
    ret[('ambe','dedx')]  = ROOT.TH1F("dedx",'',40,0.,10.)
    ret[('ambe','inclination')]  = ROOT.TH1F("inclination",'',15,0,90)
    ret[('ambe','asymmetry')]  = ROOT.TH1F("asymmetry",'',5,0,1.)
    ret[('ambe','lengthvsdistance_prof')]   = ROOT.TProfile("lengthvsdistance_prof",'',8,0,NX[GEOMETRY]/math.sqrt(2)*pixw*0.1)
    ret[('ambe','multiplicityvsdistance_prof')]   = ROOT.TProfile("multiplicityvsdistance_prof",'',12,0,NX[GEOMETRY]/math.sqrt(2)*pixw*0.1)

    ## CMOS integral variables
    ret[('ambe','cmos_integral')] = ROOT.TH1F("cmos_integral",'',50,1.0e6,2.5e6)
    ret[('ambe','cmos_mean')]     = ROOT.TH1F("cmos_mean",'',50,0.1,0.6)
    ret[('ambe','cmos_rms')]      = ROOT.TH1F("cmos_rms",'',50,1.5,3.5)

    ## PMT variables
    ret[('ambe','pmt_integral')]  = ROOT.TH1F("pmt_integral",'',100,0,150e+3)
    ret[('ambe','pmt_tot')]  = ROOT.TH1F("pmt_tot",'',100,0,400)
    ret[('ambe','pmt_density')] = ROOT.TH1F("pmt_density",'',100,0,1000)

    ## 2D vars
    ret[('ambe','integralvslength')] =  ROOT.TH2F("integralvslength",'',100,0,300*pixw,100,0,15e3)

    ret[('ambe','densityvslength')]      =  ROOT.TH2F("densityvslength",''     ,100,0,NX[GEOMETRY]*pixw,50,0,50)
    ret[('ambe','densityvslength_zoom')] =  ROOT.TH2F("densityvslength_zoom",'',50,0,1000*pixw, 50,5,50)
    ret[('ambe','calenergyvslength_zoom')] =  ROOT.TH2F("calenergyvslength_zoom",'',50,0,1000*pixw, 25,0,150)

    # ret[('fe','integralvslength')] = ROOT.TGraph()
    # ret[('fe','sigmavslength')] = ROOT.TGraph()
    # ret[('fe','sigmavsintegral')] = ROOT.TGraph()
    
    # x-axis titles
    titles = {'integral': 'I_{SC} (photons)', 'integralExt': 'I_{SC} (photons)', 'calintegral': 'E (keV)', 'calintegralExt': 'E (keV)', 'caldensity': 'density (eV/pixel)', 'dedx': 'dE/d#it{l}_{p} (keV/cm)',
              'energy': 'E (keV)', 'energyExt': 'E (keV)', 'energyFull': 'E (keV)', # these are like calintegral, but estimated in the reconstruction step
              'length':'#it{l}_{p} (mm)', 'width':'#it{w} (mm)', 'nhits': 'n_{p}', 'slimness': '#xi', 'density': '#delta (photons/pixel)',
              'cmos_integral': 'CMOS integral (photons)', 'cmos_mean': 'CMOS mean (photons)', 'cmos_rms': 'CMOS RMS (photons)',
              'pmt_integral': 'PMT integral (mV)', 'pmt_tot': 'PMT T.O.T. (ns)', 'pmt_density': 'PMT density (mV/ns)',
              'tgausssigma': '#sigma^{T}_{Gauss} (mm)', 'inclination': '#theta (deg)', 'asymmetry': 'asymmetry: (A-B)/(A+B)',
              'lengthvsdistance_prof': 'lenghthvsdistance', 'multiplicityvsdistance_prof': 'multiplicityvsdistance'}

    titles2d = {'integralvslength': ['l_{p} (mm)','photons'], 'densityvslength' : ['l_{p} (mm)','#delta (photons/pix)'],
                'densityvslength_zoom' : ['l_{p} (mm)','#delta (photons/pix)'], 'calenergyvslength_zoom' : ['l_{p} (mm)','E (keV)']}
    
    ## background histograms
    ret2 = {}
    for (region,var),h in ret.items():
        if ret[(region,var)].InheritsFrom('TH2'):
            ret[(region,var)].GetXaxis().SetTitle(titles2d[var][0])
            ret[(region,var)].GetYaxis().SetTitle(titles2d[var][1])
        elif ret[(region,var)].InheritsFrom('TH1'):
            ret[(region,var)].GetXaxis().SetTitle(titles[var])
            ret[(region,var)].GetXaxis().SetTitleSize(0.1)
        ret2[('cosm',var)] = h.Clone('cosm_{name}'.format(name=var)); ret2[('cosm',var)].Reset()
        ret2[('fe',var)]   = h.Clone('fe_{name}'.format(name=var)); ret2[('fe',var)].Reset()
        if ret[(region,var)].InheritsFrom('TH1'):
            ret[(region,var)].Sumw2()
            ret[(region,var)].SetDirectory(0)
            ret2[('cosm',var)].SetDirectory(0)
            ret2[('fe',var)].SetDirectory(0)
        else:
            ret[(region,var)].SetName(region+var)
            ret2[('cosm',var)].SetName('cosm'+var)
            ret2[('fe',var)].SetName('fe'+var)

    ret.update(ret2)

    ## now fill the histograms 
    selected = 0
    for runtype in ['fe','ambe','cosm']:
        for ie,event in enumerate(tfiles[runtype].Events):

            ## eventually, use the PMT TOT threshold to reject the cosmics
            #if getattr(event,'pmt_tot')>250:
            #    continue
            
            for cmosvar in ['cmos_integral','cmos_mean','cmos_rms']:
                ret[runtype,cmosvar].Fill(getattr(event,cmosvar))
            # for pmtvar in ['pmt_integral','pmt_tot']:
            #     ret[runtype,pmtvar].Fill(getattr(event,pmtvar))
            # ret[runtype,'pmt_density'].Fill(getattr(event,'pmt_integral')/getattr(event,'pmt_tot'))

            firstSlice = 0
            for isc in range(getattr(event,"nSc" if cluster=='sc' else 'nCl')):

                # let's do this unrolling here, before some continue breaks it
                slices = []
                if hasattr(event,"{clutype}_nslices".format(clutype=cluster)): # the Fe55 was recoed w/o that implemented (so far, since it is not needed)
                    nslices = int(getattr(event,"{clutype}_nslices".format(clutype=cluster))[isc])
                    slices = range(firstSlice,firstSlice+nslices)
                    firstSlice += nslices
                
                #if getattr(event,"{clutype}_iteration".format(clutype=cluster))[isc]!=2:
                #    continue
                nhits = getattr(event,"{clutype}_nhits".format(clutype=cluster))[isc]
                xmin = getattr(event,"{clutype}_xmin".format(clutype=cluster))[isc]
                xmax = getattr(event,"{clutype}_xmax".format(clutype=cluster))[isc]
                ymin = getattr(event,"{clutype}_ymin".format(clutype=cluster))[isc]
                ymax = getattr(event,"{clutype}_ymax".format(clutype=cluster))[isc]
                xmean = getattr(event,"{clutype}_xmean".format(clutype=cluster))[isc]
                ymean = getattr(event,"{clutype}_ymean".format(clutype=cluster))[isc]
                length =  getattr(event,"{clutype}_length".format(clutype=cluster))[isc]
                width =  getattr(event,"{clutype}_width".format(clutype=cluster))[isc]
                photons = getattr(event,"{clutype}_integral".format(clutype=cluster))[isc]
                integral = vignettingCorrection(xmean,ymean,length) * cosm_energycalib[runtype] * photons
                density = integral/nhits if nhits>0 else 0
                slimness = width/length
                gsigma =  getattr(event,"{clutype}_tgausssigma".format(clutype=cluster))[isc]*pixw

                distFromCenter = math.hypot((xmean-NX[GEOMETRY]/2),(ymean-NX[GEOMETRY]/2))*pixw

                if not limeQuietRegion(xmean,ymean):
                    continue

                # gainCalibnInt = integral*1.1 if runtype=='ambe' else integral 
                ## energy calibrated for saturation
                # calib = saturationFactorNLO(max(0,density)) # ev/ph
                calib = cosm_energycalib[runtype] * fe_calib_gauspeak[0]
                calibEnergy = integral * fe_calib_gauspeak[0] / 1000. # keV
                calibDensity  = density * fe_calib_gauspeak[0] # eV/pix

                # double loop: debugging vignetting only
                neighbors = 0
                for isc2 in range(getattr(event,"nSc" if cluster=='sc' else 'nCl')):
                    x2mean = getattr(event,"{clutype}_xmean".format(clutype=cluster))[isc2]
                    y2mean = getattr(event,"{clutype}_ymean".format(clutype=cluster))[isc2]
                    dist = math.hypot((xmean-x2mean),(ymean-y2mean))*pixw
                    if isc!=isc2 and dist < 50: # 5cm radius
                        neighbors += 1
                ret[(runtype,'multiplicityvsdistance_prof')].Fill(distFromCenter*0.1,neighbors)
                        
                ##########################
                ## SOME DEBUGGING CUTS...
                ##########################
                # if not (30e3 < gainCalibnInt < 40e+3):
                #     continue
                # if length > 1000:
                #     continue
                # if slimness < 0.3:
                #     continue
                # if density < 5:
                #     continue
                #if length>50:
                #    continue
                #if photons < 1000:
                #    continue
                # if not is60keVBkg(length,density):
                #     continue
                #if not slimnessCut(getattr(event,"{clutype}_length".format(clutype=cluster))[isc],getattr(event,"{clutype}_width".format(clutype=cluster))[isc]):
                 #   continue
                #if not integralCut(getattr(event,"{clutype}_integral".format(clutype=cluster))[isc]):
                #    continue
                # if not spotsLowDensity(length,density):
                #    continue
                #if not cosmicSelection(length,gsigma):
                #    continue
                #if not isPurpleBlob(length,density):
                #    continue

                ##########################
                ## the AmBe selection
                ##########################
                # remove long/slim cosmics
                #if length*pixw > 10. or slimness<0.4:
                #    continue
                # remove the residual low density background from pieces of cosmics not fully super-clustered
                # if density<19:
                #     continue
                # # remove the 60 keV background in AmBe and
                # if is60keVBkg(length,density):
                #     continue
                ##########################


                # ## cut with 50% sig eff and 1% bkg eff
                # if density<10:
                #     continue

                for var in ['length','width','nhits','tgausssigma']:
                    geomfact = pixw if var in ['length','width','tgausssigma'] else 1.
                    ret[(runtype,var)].Fill(geomfact * getattr(event,("{clutype}_{name}".format(clutype=cluster,name=var)))[isc])
                ret[(runtype,'lengthvsdistance_prof')].Fill(distFromCenter*0.1,length*pixw*0.1)
                ret[(runtype,'integral')].Fill(integral)
                ret[(runtype,'integralExt')].Fill(integral)
                ret[(runtype,'calintegral')].Fill(calibEnergy)
                ret[(runtype,'calintegralExt')].Fill(calibEnergy)
                # the following assumes correct saturation correction in the RECO. In LIME this is not done
                #energy_cal = getattr(event,"{clutype}_energy".format(clutype=cluster))[isc] * fe_truth / fe_calib_gauspeak[0]
                energy_cal = calibEnergy
                ret[(runtype,'energy')].Fill(energy_cal)
                ret[(runtype,'energyExt')].Fill(energy_cal)
                ret[(runtype,'energyFull')].Fill(energy_cal)
                ret[(runtype,'dedx')].Fill(energy_cal/(length*pixw/10.)) # keV/cm
                
                ret[(runtype,'slimness')].Fill(getattr(event,"{clutype}_width".format(clutype=cluster))[isc] / getattr(event,"{clutype}_length".format(clutype=cluster))[isc])
                ret[(runtype,'density')].Fill(density)
                ret[(runtype,'caldensity')].Fill(calibDensity)
                #length =  getattr(event,"{clutype}_lgausssigma".format(clutype=cluster))[isc] * pixw * 6 - sigma # subtract the sigma to remove the diffusion
                sigma =  getattr(event,"{clutype}_width".format(clutype=cluster))[isc] * pixw * 6 # use 2*3 sigma to contain 99.7% of prob.
                length_sub = math.sqrt(max(0,length*length - sigma*sigma))
                ret[(runtype,'integralvslength')].Fill(length,integral)

                ret[(runtype,'densityvslength')]     .Fill(pixw*length,density)
                ret[(runtype,'densityvslength_zoom')].Fill(pixw*length,density)
                ret[(runtype,'calenergyvslength_zoom')].Fill(pixw*length,calibEnergy)


                if length>110 and slimness<0.7:
                    xmin = getattr(event,"{clutype}_xmin".format(clutype=cluster))[isc]
                    xmax = getattr(event,"{clutype}_xmax".format(clutype=cluster))[isc]
                    ymin = getattr(event,"{clutype}_ymin".format(clutype=cluster))[isc]
                    ymax = getattr(event,"{clutype}_ymax".format(clutype=cluster))[isc]
                    ret[(runtype,'inclination')].Fill( angleWrtHorizontal(xmin,xmax,ymin,ymax) )

                    if hasattr(event,"{clutype}_nslices".format(clutype=cluster)) and len(slices)>1: # the Fe55 was recoed w/o that implemented (so far, since it is not needed)
                        # this is to check that the loop over slices is right: the sum of the slices should ~ cluster integral
                        # energy_raw = getattr(event,"{clutype}_energy".format(clutype=cluster))[isc]
                        # energy_closure = sum([getattr(event,"{clutype}_energyprof".format(clutype=cluster))[islice] for islice in slices])
                        # print ("run = {run}, event = {event}".format(run=event.run,event=event.event))
                        # print ("{runtype} ENERGY = {calint:.1f}; SUMSLICES = {mysum:.1f}".format(runtype=runtype,calint=energy_raw,mysum=energy_closure))
                        slicesA = slices[:len(slices)//2]
                        slicesB = slices[len(slices)//2:]
                        A = sum([getattr(event,"{clutype}_energyprof".format(clutype=cluster))[islice] for islice in slicesA])
                        B = sum([getattr(event,"{clutype}_energyprof".format(clutype=cluster))[islice] for islice in slicesB])
                        asymmetry = abs(A-B)/(A+B)
                        ret[(runtype,'asymmetry')].Fill( asymmetry )
                    else:
                        ret[(runtype,'asymmetry')].Fill( -1 )
                
                # ret[(runtype,'integralvslength')].SetPoint(selected,length_sub,integral) 
                # ret[(runtype,'sigmavslength')].SetPoint(selected,length_sub,sigma)
                # ret[(runtype,'sigmavsintegral')].SetPoint(selected,integral,sigma)
                selected += 1

                ### for debugging purposes:
                # if 30e3 < integral < 40e+3 and event.run==2156:
                # if 2096 < event.run < 2099 and energy_cal < 7 and density>10: # and length < 80 and 5 < density < 8:
                #     print("density = {d:.1f}\tlength = {l:.0f}\t{r}\t{e}\t{y}\t{x}\t{phot}\t{ene}".format(d=density,l=length,r=event.run,e=event.event,y=int(event.sc_ymean[isc]/4.),x=int(event.sc_xmean[isc]/4.),phot=int(event.sc_integral[isc]),ene=energy_cal))


        # energyFull has variable bin size. Normalize entries to the bin size
        for i in range(1,ret[(runtype,'energyFull')].GetNbinsX()+1):
            if ret[(runtype,'energyFull')].GetBinWidth(i)==0: print ("zero ",i)
            ret[(runtype,'energyFull')].SetBinContent(i, ret[(runtype,'energyFull')].GetBinContent(i) / ret[(runtype,'energyFull')].GetBinWidth(i))
                
    return ret,entries

def getCanvas():
    c = ROOT.TCanvas('c','',1200,1200)
    lMargin = 0.14
    rMargin = 0.10
    bMargin = 0.15
    tMargin = 0.10
    c.SetLeftMargin(lMargin)
    c.SetRightMargin(rMargin)
    c.SetTopMargin(tMargin)
    c.SetBottomMargin(bMargin)
    c.SetFrameBorderMode(0);
    c.SetBorderMode(0);
    c.SetBorderSize(0);
    return c

def drawOneProfile(profiles,var,plotdir):
    ROOT.gStyle.SetOptStat(0)
    c = getCanvas()

    miny=1e10; maxy = 0
    for i,prof in enumerate(profiles):
        prof.Draw('pe1' if i==0 else 'pe1 same')
        miny = min(prof.GetMinimum(),miny)
        maxy = max(prof.GetMaximum(),maxy)

    xtitle = var.split('vs')[1]
    ytitle = var.split('vs')[0]

    titles = [xtitle,ytitle]
    for i,t in enumerate(titles):
        if 'length' in t: titles[i] = 'length (cm)'
        if 'distance' in t: titles[i] = 'distance (cm)'
        if 'multiplicity' in t: titles[i] = 'n Sclusters in #Delta R < 5 cm'

    profiles[0].GetXaxis().SetTitle(titles[0])
    profiles[0].GetYaxis().SetTitle(titles[1])
    profiles[0].GetXaxis().SetTitleSize(0.05)
    profiles[0].GetXaxis().SetTitleFont(42)
    profiles[0].GetXaxis().SetLabelSize(0.05)
    profiles[0].GetXaxis().SetLabelFont(42)
    profiles[0].GetYaxis().SetLabelSize(0.05)
    profiles[0].GetYaxis().SetLabelFont(42)
    profiles[0].GetYaxis().SetTitleSize(0.05)
    profiles[0].GetYaxis().SetTitleFont(42)
    profiles[0].GetYaxis().SetRangeUser(miny,maxy*1.2)
    
    for ext in ['png','pdf']:
        c.SaveAs("{plotdir}/prof_{var}.{ext}".format(plotdir=plotdir,var=var,ext=ext))

    of = ROOT.TFile.Open("{plotdir}/prof_{var}.root".format(plotdir=plotdir,var=var),'recreate')
    for p in profiles:
        p.Write()
    of.Close()


def drawOneGraph(graph,var,plotdir):
    ROOT.gStyle.SetOptStat(0)
    c = ROOT.TCanvas('c','',1200,1200)
    lMargin = 0.12
    rMargin = 0.05
    bMargin = 0.30
    tMargin = 0.07

    graph.SetMarkerStyle(ROOT.kOpenCircle)
    graph.SetMarkerColor(1)
    graph.SetMarkerSize(1)
    graph.Draw('AP')

    xtitle = var.split('vs')[1]
    ytitle = var.split('vs')[0]

    if 'length' in xtitle:
        graph.GetXaxis().SetRangeUser(-6,15)
    if 'integral' in ytitle:
        graph.GetYaxis().SetRangeUser(0,100)
    if 'integral' in xtitle:
        graph.GetXaxis().SetRangeUser(0,100)
        
    titles = [xtitle,ytitle]
    for i,t in enumerate(titles):
        if 'integral' in t:  titles[i] = 'energy [keV]'
        if 'sigma' in t: titles[i] = 'transverse width (6#sigma) [mm]'
        if 'length' in t: titles[i] = 'length (diffusion subtracted) [mm]'

    graph.GetXaxis().SetTitle(titles[0])
    graph.GetYaxis().SetTitle(titles[1])

    for ext in ['png','pdf','root']:
        c.SaveAs("{plotdir}/graph_{var}.{ext}".format(plotdir=plotdir,var=var,ext=ext))


def drawOne(histo_sig,histo_bkg,histo_sig2=None,plotdir='./',normEntries=False):
    ROOT.gStyle.SetOptStat(0)

    c = ROOT.TCanvas('c','',1200,1200)
    lMargin = 0.12
    rMargin = 0.05
    bMargin = 0.30
    tMargin = 0.07
    padTop = ROOT.TPad('padTop','',0.,0.4,1,0.98)
    padTop.SetLeftMargin(lMargin)
    padTop.SetRightMargin(rMargin)
    padTop.SetTopMargin(tMargin)
    padTop.SetBottomMargin(0)
    padTop.SetFrameBorderMode(0);
    padTop.SetBorderMode(0);
    padTop.SetBorderSize(0);
    padTop.Draw()

    padBottom = ROOT.TPad('padBottom','',0.,0.02,1,0.4)
    padBottom.SetLeftMargin(lMargin)
    padBottom.SetRightMargin(rMargin)
    padBottom.SetTopMargin(0)
    padBottom.SetBottomMargin(bMargin)
    padBottom.SetFrameBorderMode(0);
    padBottom.SetBorderMode(0);
    padBottom.SetBorderSize(0);
    padBottom.Draw()

    padTop.cd()
    ymax = max(histo_bkg.GetMaximum(),histo_sig.GetMaximum())
    if histo_sig2:
        ymax = max(histo_sig2.GetMaximum(),ymax)
    histo_sig.SetMaximum(1.4*ymax)
    #histo_sig.SetMinimum(0)
    histo_sig.GetXaxis().SetLabelSize(0.05)
    histo_sig.GetXaxis().SetLabelFont(42)
    histo_sig.GetYaxis().SetLabelSize(0.05)
    histo_sig.GetYaxis().SetLabelFont(42)
    histo_sig.GetYaxis().SetTitleSize(0.05)
    histo_sig.GetYaxis().SetTitleFont(42)
    if normEntries:
        binNorm = ' / bin size ' if histo_sig.GetName()=='energyFull' else ' '
        histo_sig.GetYaxis().SetTitle('clusters{norm}(normalized to AmBe)'.format(norm=binNorm))
    else:
        histo_sig.GetYaxis().SetTitle('clusters (a.u.)')
    histo_sig.Draw("pe")
    histo_bkg.Draw("hist same")

    if histo_sig2:
        histo_sig2.Draw("hist same")
    if histo_sig.GetName()=='energyFull':
        histo_sig.SetMaximum(1e3)
        #histo_sig.SetMinimum(5e-4)
        padTop.SetLogy(1)

    ### this is to fit the energy/length distribution in the cosmics
    # if histo_sig.GetName()=='dedx':
    #     l1 = ROOT.TF1("l1","landau",2,9);
    #     histo_bkg.Fit('l1','S')
    #     l1.Draw('same')
    #     mpv  = l1.GetParameter(1); mErr = l1.GetParError(1)
    #     sigma = l1.GetParameter(2); mSigma = l1.GetParError(2)
    #     lat1 = ROOT.TLatex()
    #     lat1.SetNDC(); lat1.SetTextFont(42); lat1.SetTextSize(0.05)
    #     lat1.DrawLatex(0.55, 0.60, "mpv = {m:.2f} #pm {em:.2f} keV".format(m=mpv,em=mErr))
    #     lat1.DrawLatex(0.55, 0.50, "#sigma = {s:.2f} #pm {es:.2f} keV".format(s=sigma,es=mSigma))
    ###############################

    ## rescale the Fe bkg by the scale factor in the pure cosmics CR
    histo_bkg.Scale(cosm_rate_calib[0])
    
    histos = [histo_sig,histo_bkg] + ([histo_sig2] if histo_sig2 else [])
    labels = ['AmBe','%.2f #times no source' % cosm_rate_calib[0]] + ([ '%.2f #times ^{55}Fe' % fe_integral_rescale] if histo_sig2 else [])
    styles = ['pe','f'] + (['f'] if histo_sig2 else [])
    
    legend = doLegend(histos,labels,styles,corner="TR")
    legend.Draw()


    
    padBottom.cd()
    ratios = []; labelsR = []; stylesR = []
    ratio = histo_sig.Clone(histo_sig.GetName()+"_diff")
    ratio.SetMarkerStyle(ROOT.kFullDotLarge)
    ratio.SetMarkerColor(ROOT.kBlack)
    ratio.SetLineColor(ROOT.kBlack)
    ratio.Sumw2()
    ratio.Add(histo_bkg,-1.0)
    ratio.GetXaxis().SetLabelSize(0.05 * 3./2.)
    ratio.GetXaxis().SetLabelFont(42)
    ratio.GetYaxis().SetLabelSize(0.05 * 3./2.)
    ratio.GetYaxis().SetLabelFont(42)
    ratio.GetYaxis().SetTitleSize(0.05 * 3./2.)
    ratio.GetYaxis().SetTitleFont(42)
    ratio.GetYaxis().SetTitleOffset(0.8)
    ratio.GetYaxis().SetTitle("bkg. subtr. data")
    ratio.GetYaxis().CenterTitle()
    # take the first error bar as estimate for all)
    ratio.SetMaximum(1.4*ratio.GetMaximum()+ratio.GetBinError(1))
    #ratio.SetMinimum(0)
    ratio.Draw('pe')
    
    if histo_sig.GetName()=='energyFull':
        ratio.SetMaximum(1e3)
        #ratio.SetMinimum(8e-4)
        padBottom.SetLogy(1)

    ratios.append(ratio)
    labelsR.append('AmBe - no source')
    stylesR.append('pe')

    ## bad hack... Just fit the distribution for the calib integral if selecting the 60 keV structure
    ## in principle should fit the bkg-subtracted plot, but too large stat uncertainty on BKG
    if histo_sig.GetName() in ['calintegralExt','energyExt']:
        g1 = ROOT.TF1("g1","gaus",20,110);
        ratio.Fit('g1','RS')
        mean  = g1.GetParameter(1); mErr = g1.GetParError(1)
        sigma = g1.GetParameter(2); mSigma = g1.GetParError(2)
        lat = ROOT.TLatex()
        lat.SetNDC(); lat.SetTextFont(42); lat.SetTextSize(0.07)
        lat.DrawLatex(0.65, 0.60, "mean = {m:.1f} #pm {em:.1f} keV".format(m=mean,em=mErr))
        lat.DrawLatex(0.65, 0.50, "#sigma = {s:.1f} #pm {es:.1f} keV".format(s=sigma,es=mSigma))
    ###############################
    
    
    if histo_sig2:
        ratio2 = histo_sig2.Clone(histo_sig.GetName()+"fe_diff")
        ratio2.SetMarkerStyle(ROOT.kFullDotLarge)
        ratio2.SetMarkerColor(ROOT.kRed+2)
        ratio2.SetLineColor(ROOT.kRed+2)
        ratio2.Sumw2()
        ratio2.Add(histo_bkg,-1.0)
        ratio2.GetYaxis().SetTitleSize(0.05)
        ratio2.GetYaxis().SetTitle("{num} - {den}".format(num=labels[0],den=labels[1]))
        ratio2.Draw('pe same')
        ratios.append(ratio2)
        labelsR.append('%.1f #times ^{55}Fe - no source' % fe_integral_rescale)
        stylesR.append('pe')
        rmax = max(ratio.GetMaximum(),ratio2.GetMaximum())
        ratio.SetMaximum(1.1*rmax)
        
    legendR = doLegend(ratios,labelsR,stylesR,corner="TR")
    legendR.Draw()

    line = ROOT.TLine()
    line.DrawLine(ratio.GetXaxis().GetBinLowEdge(1), 0, ratio.GetXaxis().GetBinLowEdge(ratio.GetNbinsX()+1), 0)
    line.SetLineStyle(3)
    line.SetLineColor(ROOT.kBlack)
    
    for ext in ['png','pdf']:
        c.SaveAs("{plotdir}/{var}.{ext}".format(plotdir=plotdir,var=histo_sig.GetName(),ext=ext))

    of = ROOT.TFile.Open("{plotdir}/{var}.root".format(plotdir=plotdir,var=histo_sig.GetName()),'recreate')
    histo_bkg.Write()
    histo_sig.Write()
    ratio.Write()
    if histo_sig2:
        histo_sig2.Write()
        ratio2.Write()
    of.Close()

## this just plots 2D in the dumbest way
def drawOne2D_raw(histo_sig,histo_bkg1,histo_bkg2,plotdir='./'):
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetPalette(ROOT.kRainBow)
    
    ROOT.gStyle.SetNumberContours(51)
    ROOT.gErrorIgnoreLevel = 100
    
    c = ROOT.TCanvas('c','',3600,1200)
    c.Divide(3,1)

    c.cd(1)
    histo_sig.SetTitle('AmBe')
    histo_sig.Draw('colz')
    
    c.cd(2)
    histo_bkg1.SetTitle('^{55}Fe')
    histo_bkg1.Draw("colz")

    c.cd(3)
    histo_bkg2.SetTitle('no source')
    histo_bkg2.Draw("colz")

    
    for ext in ['png','pdf']:
        c.SaveAs("{plotdir}/{var}_3species.{ext}".format(plotdir=plotdir,var=histo_sig.GetName(),ext=ext))

    of = ROOT.TFile.Open("{plotdir}/{var}_3species.root".format(plotdir=plotdir,var=histo_sig.GetName()),'recreate')
    histo_bkg2.Write()
    histo_bkg1.Write()
    histo_sig.Write()
    of.Close()
    

## the following does background subtraction with a second histogram
def drawOne2D(histo_sig,histo_bkg,plotdir='./',normEntries=False):
    ROOT.gStyle.SetOptStat(0)

    # ROOT.TColor.CreateGradientColorTable(3,
    #                                   array ("d", [0.00, 0.50, 1.00]),
    #                                   ##array ("d", [1.00, 1.00, 0.00]),
    #                                   ##array ("d", [0.70, 1.00, 0.34]),
    #                                   ##array ("d", [0.00, 1.00, 0.82]),
    #                                   array ("d", [0.00, 1.00, 1.00]),
    #                                   array ("d", [0.34, 1.00, 0.65]),
    #                                   array ("d", [0.82, 1.00, 0.00]),
    #                                   255,  0.95)

    ROOT.gStyle.SetPalette(ROOT.kRainBow)
    #ROOT.gStyle.SetNumberContours(51)
    ROOT.gErrorIgnoreLevel = 100
    
    c = ROOT.TCanvas('c','',3600,1200)
    c.Divide(3,1)

    c.cd(1)
    histo_sig.SetTitle('(AmBe)')
    histo_sig.Draw('colz')
    
    c.cd(2)
    histo_bkg.SetTitle('(no source)')
    histo_bkg.SetMaximum(30)
    histo_bkg.Draw("colz")

    c.cd(3)
    ratio = histo_sig.Clone(histo_sig.GetName()+"_diff")
    ratio.SetTitle('(AmBe) - %.2f #times (no source)' % cosm_rate_calib[0])
    ratio.Sumw2()
    ratio.Add(histo_bkg,-1*cosm_rate_calib[0])
    #ratio.GetYaxis().SetTitleSize(0.05)
    ratio.GetYaxis().SetTitle(histo_sig.GetYaxis().GetTitle())
    ratio.Draw('colz')
    
    maxY = max(ratio.GetMaximum(),abs(ratio.GetMinimum()))
    ratio.SetMaximum( min(0.5 * maxY,15))
    ratio.SetMinimum(0)
    #ratio.SetMinimum(-0.5 * maxY)
    
    for ext in ['png','pdf']:
        c.SaveAs("{plotdir}/{var}.{ext}".format(plotdir=plotdir,var=histo_sig.GetName(),ext=ext))

    of = ROOT.TFile.Open("{plotdir}/{var}.root".format(plotdir=plotdir,var=histo_sig.GetName()),'recreate')
    histo_bkg.Write()
    histo_sig.Write()
    of.Close()

def drawSpectra(histos,plotdir,entries,normEntries=False):
    variables = [var for (reg,var) in list(histos.keys()) if reg=='fe']

    ROOT.gStyle.SetHatchesSpacing(0.3)
    ROOT.gStyle.SetHatchesLineWidth(1)

    for var in variables:
        if histos[('ambe',var)].InheritsFrom('TH1') and not histos[('ambe',var)].InheritsFrom('TProfile'):
            if normEntries:
                histos[('fe',var)].Scale(float(entries['ambe'])/float(entries['fe'])*fe_integral_rescale)
                histos[('cosm',var)].Scale(float(entries['ambe'])/float(entries['cosm']))
            else:
                histos[('ambe',var)].Scale(1./histos[('ambe',var)].Integral())
                histos[('cosm',var)].Scale(1./histos[('cosm',var)].Integral())
                histos[('fe',var)].Scale(1./histos[('fe',var)].Integral())
        if histos[('ambe',var)].InheritsFrom('TH2'):
            if var in ['integralvslength','densityvslength_zoom','caldensityvslength_zoom']:
                drawOne2D(histos[('ambe',var)],histos[('fe',var)],plotdir)
            drawOne2D_raw(histos[('ambe',var)],histos[('fe',var)],histos[('cosm',var)],plotdir)
        elif histos[('fe',var)].InheritsFrom('TProfile'):
            histos[('ambe',var)].SetMarkerStyle(ROOT.kFullDotLarge)
            histos[('ambe',var)].SetLineColor(ROOT.kBlack)
            histos[('cosm',var)].SetMarkerStyle(ROOT.kOpenCircle)
            histos[('cosm',var)].SetLineColor(ROOT.kGreen+3)
            drawOneProfile([histos[('ambe',var)],histos[('cosm',var)]],var,plotdir)
        elif histos[('ambe',var)].InheritsFrom('TH1'):
            histos[('ambe',var)].SetMarkerStyle(ROOT.kFullDotLarge)
            histos[('ambe',var)].SetLineColor(ROOT.kBlack)
            histos[('cosm',var)].SetFillColorAlpha(ROOT.kGreen-5,0.3)
            #histos[('cosm',var)].SetFillStyle(3345)
            if histos[('fe',var)]:
                histos[('fe',var)].SetFillColorAlpha(ROOT.kViolet-4,0.5)
                #histos[('fe',var)].SetFillStyle(3354)
            drawOne(histos[('ambe',var)],histos[('cosm',var)],histos[('fe',var)],plotdir,normEntries)
            #drawOne(histos[('ambe',var)],histos[('cosm',var)],histo_sig2=None,plotdir=plotdir,normEntries=normEntries)
        elif histos[('fe',var)].InheritsFrom('TGraph'):
            drawOneGraph(histos[('ambe',var)],var,plotdir)
            
def plotEnergyVsDistance(plotdir):

    tf_fe55 = ROOT.TFile('runs/reco_run01740_3D.root')
    tree = tf_fe55.Get('Events')

    np = 9 # number of points
    dist = 100 # distance from center in pixels
    x = [dist*i for i in range(np+1)]
    y_mean = []; y_res = []

    cut_base = 'cl_iteration==2'
    integral = ROOT.TH1F('integral','',100,600,3000)
    
    for p in range(np):
        cut = "{base} && TMath::Hypot(cl_xmean-1024,cl_ymean-1024)>{r_min} && TMath::Hypot(cl_xmean-1024,cl_ymean-1024)<{r_max}".format(base=cut_base,r_min=x[p],r_max=x[p+1])
        tree.Draw('cl_integral>>integral',cut)
        mean = integral.GetMean()
        rms  = integral.GetRMS()
        y_mean.append(mean)
        y_res.append(rms/mean)
        integral.Reset()

    gr_mean = ROOT.TGraph(np,array('f',x),array('f',y_mean))
    gr_res = ROOT.TGraph(np,array('f',x),array('f',y_res))
    gr_mean.SetMarkerStyle(ROOT.kOpenCircle)
    gr_res.SetMarkerStyle(ROOT.kOpenCircle)
    gr_mean.SetMarkerSize(2)
    gr_res.SetMarkerSize(2)
    
    gr_mean.SetTitle('')
    gr_res.SetTitle('')
    
    c = ROOT.TCanvas('c','',1200,1200)
    lMargin = 0.12
    rMargin = 0.05
    bMargin = 0.30
    tMargin = 0.07

    gr_mean.Draw('AP')
    gr_mean.GetXaxis().SetTitle('distance from center (pixels)')
    gr_mean.GetYaxis().SetTitle('integral (photons)')

    for ext in ['png','pdf']:
        c.SaveAs("{plotdir}/mean.{ext}".format(plotdir=plotdir,ext=ext))

    gr_res.Draw('AP')
    gr_res.GetXaxis().SetTitle('distance from center (pixels)')
    gr_res.GetYaxis().SetTitle('resolution (rms)')
    gr_res.GetYaxis().SetRangeUser(0.10,0.50)

    for ext in ['png','pdf']:
        c.SaveAs("{plotdir}/rms.{ext}".format(plotdir=plotdir,ext=ext))

    print(x)
    print(y_mean)
    print(y_res)

def plotCameraEnergyVsPosition(plotdir,var='integral'):
    
    x = []
    y_mean = []; y_res = []
    
    for i in range(0,6):
        
        mean, rms, i, leg = plotHistFit(plotdir,var,i)
        x.append(i)
        y_mean.append(mean)
        y_res.append(rms/mean)

    print(y_res)
    np = len(x)
    
    gr_mean = ROOT.TGraph(np,array('f',x),array('f',y_mean))
    gr_res = ROOT.TGraph(np,array('f',x),array('f',y_res))
    gr_mean.GetXaxis().SetRangeUser(0,21)
    gr_res.GetXaxis().SetRangeUser(0,21)
    gr_mean.GetYaxis().SetRangeUser(0,max(y_mean)*1.2)
    gr_res. GetYaxis().SetRangeUser(0,max(y_res)*1.2)
    gr_mean.SetMarkerStyle(ROOT.kOpenCircle)
    gr_res.SetMarkerStyle(ROOT.kOpenCircle)
    gr_mean.SetMarkerSize(2)
    gr_res.SetMarkerSize(2)
    
    gr_mean.SetTitle('')
    gr_res.SetTitle('')

    c = ROOT.TCanvas('c','',1200,1200)
    lMargin = 0.17
    rMargin = 0.05
    bMargin = 0.15
    tMargin = 0.07
    c.SetLeftMargin(lMargin)
    c.SetRightMargin(rMargin)
    c.SetTopMargin(tMargin)
    c.SetBottomMargin(bMargin)
    c.SetFrameBorderMode(0);
    c.SetBorderMode(0);
    c.SetBorderSize(0);
    c.SetGrid();

    gr_mean.Draw('AP')
    gr_mean.GetXaxis().SetTitle('Source Position [cm]')
    gr_mean.GetYaxis().SetTitle('{leg}'.format(leg=leg))

    for ext in ['png','pdf','root']:
        c.SaveAs("{plotdir}/CameraMean_{var}_60-40.{ext}".format(plotdir=plotdir,var=var,ext=ext))

    gr_res.Draw('AP')
    gr_res.GetXaxis().SetTitle('Source Position [cm]')
    gr_res.GetYaxis().SetTitle('Resolution [rms]')

    for ext in ['png','pdf','root']:
        c.SaveAs("{plotdir}/CameraRMS_{var}_60-40.{ext}".format(plotdir=plotdir,var=var,ext=ext))
    
    
def plotPMTEnergyVsPosition(plotdir):
    tf_fe55 = ROOT.TFile('runs/reco_run01754_to_run01759.root')
    tree = tf_fe55.Get('Events')

    runs = list(range(1754,1760))
    integral = ROOT.TH1F('integral','',100,2000,20000)

    np = len(runs)
    x = []
    y_mean = []; y_res = []
    
    for i,r in enumerate(runs):
        cut = 'run=={r} && pmt_tot<100'.format(r=r)
        tree.Draw('pmt_integral>>integral',cut)
        mean = integral.GetMean()
        rms  = integral.GetRMS()
        x.append(i)
        y_mean.append(mean)
        y_res.append(rms/mean)
        integral.Reset()


    print(y_res)
    
    gr_mean = ROOT.TGraph(np,array('f',x),array('f',y_mean))
    gr_res = ROOT.TGraph(np,array('f',x),array('f',y_res))
    gr_mean.GetXaxis().SetRangeUser(-1,np)
    gr_res.GetXaxis().SetRangeUser(-1,np)
    gr_mean.GetYaxis().SetRangeUser(5000,9000)
    gr_res. GetYaxis().SetRangeUser(0,1.0)
    gr_mean.SetMarkerStyle(ROOT.kOpenCircle)
    gr_res.SetMarkerStyle(ROOT.kOpenCircle)
    gr_mean.SetMarkerSize(2)
    gr_res.SetMarkerSize(2)
    
    gr_mean.SetTitle('')
    gr_res.SetTitle('')

    c = ROOT.TCanvas('c','',1200,1200)
    lMargin = 0.17
    rMargin = 0.05
    bMargin = 0.15
    tMargin = 0.07
    c.SetLeftMargin(lMargin)
    c.SetRightMargin(rMargin)
    c.SetTopMargin(tMargin)
    c.SetBottomMargin(bMargin)
    c.SetFrameBorderMode(0);
    c.SetBorderMode(0);
    c.SetBorderSize(0);
    

    gr_mean.Draw('AP')
    gr_mean.GetXaxis().SetTitle('source position index')
    gr_mean.GetYaxis().SetTitle('PMT integral (mV)')

    for ext in ['png','pdf']:
        c.SaveAs("{plotdir}/mean.{ext}".format(plotdir=plotdir,ext=ext))

    gr_res.Draw('AP')
    gr_res.GetXaxis().SetTitle('source position index')
    gr_res.GetYaxis().SetTitle('resolution (rms)')

    for ext in ['png','pdf']:
        c.SaveAs("{plotdir}/rms.{ext}".format(plotdir=plotdir,ext=ext))

    print(x)
    print(y_mean)
    print(y_res)


def plotCameraPMTCorr(outdir):
    tf_fe55 = ROOT.TFile('runs/reco_run01754_to_run01759.root')
    tree = tf_fe55.Get('Events')

    tot_vs_nhits = ROOT.TH2F('nhits_vs_tot','',45,20,100,30,100,400)
    tot_vs_nhits.GetXaxis().SetTitle("T.o.T. (ms)")
    tot_vs_nhits.GetYaxis().SetTitle("supercluster pixels")
    tot_vs_nhits.SetContour(100)
    
    ## fill the 2D histogram
    for event in tree:
        if event.pmt_tot > 100: continue
        for isc in range(event.nSc):
            if event.sc_iteration[isc]!=2: continue
            if event.sc_width[isc]/event.sc_length[isc]<0.7: continue
            if not withinFC(event.sc_xmean[isc],event.sc_ymean[isc],700,700): continue
            tot_vs_nhits.Fill(event.pmt_tot,event.sc_nhits[isc])

    ## profile for better understanding
    profX = tot_vs_nhits.ProfileX()
    profX.SetMarkerStyle(ROOT.kFullCircle)
    profX.SetLineColor(ROOT.kBlack)
    profX.SetMarkerColor(ROOT.kBlack)
    profX.GetYaxis().SetRangeUser(180,310)
    profX.GetYaxis().SetTitle("average pixels in SC")

    ROOT.gStyle.SetPalette(ROOT.kRainBow)
    ROOT.gStyle.SetOptStat(0)
    c = getCanvas()
    tot_vs_nhits.Draw("colz")
    for ext in ['png','pdf','root']:
        c.SaveAs('{plotdir}/{name}.{ext}'.format(plotdir=outdir,name=tot_vs_nhits.GetName(),ext=ext))

    profX.Draw('pe1')
    for ext in ['png','pdf','root']:
        c.SaveAs('{plotdir}/{name}_profX.{ext}'.format(plotdir=outdir,name=tot_vs_nhits.GetName(),ext=ext))
        
def plotHistFit(plotdir,var='integral',i=0):
    import numpy as np
    
    ROOT.gStyle.SetOptFit(1011)
    gas=60
    if gas == 70:
        pos  = [0, 1, 2, 3, 4, 5, 6]
        run  = [2274, 2275, 2276, 2277, 2278, 2279, 2280]
        dist = (23-np.array([19.5, 16.5, 14.3, 12.5, 10.5, 8.2, 6.2])).tolist()
    else:
        pos  = [0, 1, 2, 3, 4, 5]
        run  = [2160, 2161, 2162, 2163, 2164, 2165]
        dist = (23-np.array([19.5, 16.5, 14.3, 12.5, 10.5, 8.2])).tolist()
        
    
    if var == 'integral':
        var1 = 'sc_integral'
        leg = 'Integral [ph]'
        histlimit = 6000
    elif var == 'length':
        var1 = 'sc_length*125E-3'
        leg = 'Length [mm]'
        if gas == 70:
            histlimit = 12
        else:
            histlimit = 7
    elif var == 'width':
        var1 = 'sc_width*125E-3'
        leg = 'Width [mm]'
        histlimit = 10
    elif var == 'size':
        var1 = 'sc_size'
        leg = 'supercluster area [px]'
        histlimit = 1600
    elif var == 'nhits':
        var1 = 'sc_nhits'
        leg = 'Supercluster active pixels [px]'
        histlimit = 1600
    elif var == 'slimness':
        var1 = 'sc_width/sc_length'
        leg = 'Slimness [w/l]'
        histlimit = 1.2
    elif var == 'density':
        var1 = 'sc_integral/sc_nhits'
        leg = 'Density [photons/pix]'
        histlimit = 15
    elif var == 'energy':
        var1 = 'calibratedEnergy(sc_integral,sc_integral/sc_nhits)'
        print ("var1  = ",var1)
        leg = 'calibrated energy (keV)'
        histlimit = 20
    else:
        exit()
    
    if gas == 70:
        tf_fe55 = ROOT.TFile('../reco_run02274_to_run02280.root')    
    else:
        tf_fe55 = ROOT.TFile('../runs/AmBeConfigCalib/reco_runs_Fe55_ZScan6040_3D.root')
        
    tree = tf_fe55.Get('Events')
    
    c = ROOT.TCanvas('','',800,600)

    hist = ROOT.TH1F('hist','%.2f cm between Source and GEM' % (dist[i]),70,0,histlimit)
    hist.Sumw2()

    cut_base = 'run=={r}'.format(r=run[i])
    cut = "{base} && TMath::Hypot(sc_xmean-1024,(sc_ymean-1024)*1.2)<{r_max}".format(base=cut_base,r_max=400)
    ## for the Fe55, remove long and slim tracks
    cut = "{cut} && sc_length<500".format(cut=cut)
    
    tree.Draw("{var}>>hist".format(var=var1),cut)
    hist.SetFillStyle(3005)
    hist.SetMarkerStyle(ROOT.kFullCircle)
    hist.SetMarkerSize(2)
    hist.SetLineColor(ROOT.kBlack)
    hist.SetMarkerColor(ROOT.kBlack)
    mean = hist.GetMean()
    rms  = hist.GetRMS()
    
    # add Polya Fuction
    func='gauss'
    
    if func == 'gauss':
        print("Using Gauss fit")
        f = ROOT.TF1('f','gaus',mean-rms,mean+rms) # there is a tail, so +/- 1sigma is sufficient to fit the core
        f.SetParameter(1,mean);
        f.SetParLimits(1,mean-2*rms,mean+2*rms);
        f.SetParameter(2,rms/2.); # there is a tail
        #f.SetParLimits(2,300,600);
        fitr_xmin = mean-rms
        fitr_xmax = mean+rms
        if var == 'density':
            fitr_xmin = 6 if i==0 else 8
            fitr_xmax = 11
        fitRe = hist.Fit(f,'S','',fitr_xmin,fitr_xmax)
        rMean  = f.GetParameter(1)
        rSigma = f.GetParameter(2)
    else:
        print("Using Polya fit needs to be fixed")
             
        #[0] = b
        #[1] = nt
        #[2] = k
        #k = 1/[0] -1
        #[x] = n

        pol_A = "(1/[0]*[1])"
        pol_B = "(1/TMath::Factorial(1/([0] -1)))"
        pol_C = "TMath::Power(x/([0]*[1]), (1/([0] -1)))"
        pol_D = "exp((-1*x)/([0]*[1]))"
        pol = "(%s)*(%s)*(%s)*(%s)" % (pol_A , pol_B , pol_C, pol_D)
        
        polyaFit = ROOT.TF1("polyafit", pol, 0, 6000)
        #polyaFit.SetParameter (0, 500)
        polyaFit.SetParameter (1, 2600)
        
        hist.Fit(polyaFit ,"S")
        rMean  = polyaFit.GetParameter(0)
        rSigma = polyaFit.GetParameter(1)
        
    hist.Draw('pe1 sames')  
    
    ROOT.gPad.Update()
    #h.GetXaxis().SetRangeUser(0,0.25)
    hist.GetYaxis().SetTitle('Counts')
    hist.GetXaxis().SetTitle('{leg}'.format(leg=leg))
    #'Position %d' % (pos[i])
    c.SetGrid()
    c.Draw()
    for ext in ['png','pdf','root']:
        c.SaveAs("{plotdir}/hist_{var}_pos{pos}_60-40.{ext}".format(plotdir=plotdir,var=var,pos=pos[i],ext=ext))
    hist.Reset()
    
    return rMean,rSigma,dist[i],leg

def plotHist2D(plotdir,v1='integral',v2='slimness',i=0):
    import numpy as np
    
    ROOT.gStyle.SetOptFit(1011)
    
    gas=70
    #i=5
    
    if gas == 70:
        pos  = [0, 1, 2, 3, 4, 5, 6]
        run  = [2274, 2275, 2276, 2277, 2278, 2279, 2280]
        dist = (23-np.array([19.5, 16.5, 14.3, 12.5, 10.5, 8.2, 6.2])).tolist()
        gg   = '70-30'
    else:
        pos  = [0, 1, 2, 3, 4, 5]
        run  = [2160, 2161, 2162, 2163, 2164, 2165]
        dist = (23-np.array([19.5, 16.5, 14.3, 12.5, 10.5, 8.2])).tolist()
        gg   = '60-40'
        
    if gas == 70:
        tf_fe55 = ROOT.TFile('../reco_run02274_to_run02280.root')    
    else:
        tf_fe55 = ROOT.TFile('../reco_run02160_to_run02165.root')        
    
    vr1,legy,histlimity = varChoice(v1)
    vr2,legx,histlimitx = varChoice(v2)
        
    var1 = vr1+":"+vr2
        
    tree = tf_fe55.Get('Events')
    
    c = ROOT.TCanvas('','',800,800)

    hist = ROOT.TH2F('hist2D','%.2f cm between Source and GEM' % (dist[i]),1000,0,histlimity,1000,0,histlimitx)
    hist.Sumw2()

    cut_base = 'cl_iteration==2 && run=={r}'.format(r=run[i])
    cut = "{base} && TMath::Hypot(cl_xmean-1024,(cl_ymean-1024)*1.2)<{r_max}".format(base=cut_base,r_max=400)

    tree.Draw("{var}>>hist".format(var=var1),cut)
    hist.SetFillStyle(3005)
    
    ROOT.gPad.Update()
    #h.GetXaxis().SetRangeUser(0,0.25)
    hist.GetYaxis().SetTitle('{leg}'.format(leg=legy))
    hist.GetXaxis().SetTitle('{leg}'.format(leg=legx))
    c.SetGrid()
    c.Draw()
    c.SaveAs("{plotdir}/hist_{var}_vs_{var2}_pos{pos}_{gg}.pdf".format(plotdir=plotdir,var=v1,var2=v2,pos=pos[i],gg=gg))
    hist.Reset()


def getOneROC(sigh,bkgh,direction='gt',verbose=False):
    
    ## this assumes the same binning, but it is always the case here
    plots = [sigh,bkgh]
    integrals = [p.Integral() for p in plots]
    nbins = sigh.GetNbinsX()

    efficiencies = []
    for ip,p in enumerate(plots):
        if direction=='gt':
            eff = [p.Integral(binx,nbins)/integrals[ip] for binx in range(0,nbins)]
            thr = [p.GetBinLowEdge(binx) for binx in range(0,nbins)]
        else:
            eff = [p.Integral(0,binx)/integrals[ip] for binx in range(0,nbins)]
            thr = [p.GetBinLowEdge(binx+1) for binx in range(0,nbins)]
        # this is to find the cut for the wanted eff
        effdic = dict(zip(thr,eff))
        if verbose: print (effdic)
        efficiencies.append(eff)

    ## graph for the roc
    roc = ROOT.TGraph(nbins)
    for i in range(nbins):
        roc.SetPoint(i, efficiencies[0][i], 1-efficiencies[1][i])
        if verbose: print ("point {i}, S eff={seff:.2f}, B eff={beff:.3f}".format(i=i,seff=efficiencies[0][i],beff=efficiencies[1][i]))
    roc.SetTitle('')
        
    return roc

def drawROC(varname,odir):
    tf = ROOT.TFile.Open('{odir}/{var}.root'.format(odir=odir,var=varname))

    print ("===> Cosm roc")
    cosm_roc = getOneROC(tf.Get('{var}_diff'.format(var=varname)), tf.Get('cosm_{var}'.format(var=varname)))
    print ("===> Fe roc")
    fe_roc   = getOneROC(tf.Get('{var}_diff'.format(var=varname)), tf.Get('fe_{var}'.format(var=varname)))

    c = ROOT.TCanvas('c','',1200,1200)
    cosm_roc.SetMarkerStyle(ROOT.kFullDotLarge)
    cosm_roc.SetMarkerSize(2)
    cosm_roc.SetMarkerColor(ROOT.kBlack)
    cosm_roc.SetLineColor(ROOT.kBlack)

    fe_roc.SetMarkerStyle(ROOT.kFullSquare)
    fe_roc.SetMarkerSize(2)
    fe_roc.SetMarkerColor(ROOT.kRed)
    fe_roc.SetLineColor(ROOT.kRed)

    fe_roc.Draw('APC')
    fe_roc.GetXaxis().SetTitle('signal efficiency')
    fe_roc.GetYaxis().SetTitle('Background rejection')
    #cosm_roc.Draw('PC')

    graphs = [cosm_roc,fe_roc]
    labels = ['no source bkg','^{55}Fe bkg']
    styles = ['pl','pl']
    legend = doLegend(graphs[1:],labels[1:],styles[1:],corner="TR")
    
    for ext in ['png','pdf']:
        c.SaveAs("{plotdir}/{var}_roc.{ext}".format(plotdir=odir,var=varname,ext=ext))

    fout = ROOT.TFile.Open("{plotdir}/{var}_roc.root".format(plotdir=odir,var=varname),'recreate')
    fe_roc.Write()
    fout.Close()
    

def plotPedRMS():
    rf = ROOT.TFile.Open('../pedestals/pedmap_run2109_rebin1.root')
    histo_rms = rf.Get('pedrms')
    canv = getCanvas()

    ROOT.gStyle.SetOptStat(1100)
    
    histo_rms.GetXaxis().SetTitle("electronics noise (RMS counts)")
    histo_rms.GetYaxis().SetTitle("Number of pixels")
    histo_rms.GetXaxis().SetTitleOffset(1.4)
    histo_rms.GetYaxis().SetTitleOffset(2.0)
    histo_rms.GetYaxis().SetTitleFont(42)
    histo_rms.GetXaxis().SetTitleFont(42)
    histo_rms.SetTitle("")
    
    histo_rms.Draw()
    ROOT.gPad.Update()
    
    stats = histo_rms.FindObject("stats")
    stats.SetX1NDC(0.7); stats.SetX2NDC(0.9);
    stats.SetY1NDC(0.7); stats.SetY2NDC(0.9);

    canv.SaveAs("sensor_noise.pdf")

def plotSimEoverEtrue():

    xmin=0.3
    xmax=1.1
    nbins = 70
    
    work = ROOT.RooWorkspace()
    work.factory('CBShape::cb(x[{xmin},{xmax}],mean[0.5,1.1],sigma[0.05,0.01,0.10],alpha[1,0.1,10],n[5,1,10])'.format(xmin=xmin,xmax=xmax))
    work.Print()
    
    x = work.var('x')

    var = 'sc_integral*6/3000/6'

    ms_nonoise = ROOT.kOpenSquare
    ms_noise = ROOT.kFullCircle
    lc_nonoise = ROOT.kAzure-6
    lc_noise = ROOT.kOrange-3
    
    rf_puremc = ROOT.TFile.Open("../sim/Reco_Digi_He_6_kev_petrucci.root")
    t_puremc = rf_puremc.Get('Events')
    histo_nonoise = ROOT.TH1F('histo_nonoise','',nbins,xmin,xmax)
    histo_nonoise.SetMarkerStyle(ms_nonoise)
    histo_nonoise.SetLineColor(ROOT.kBlack)
    t_puremc.Draw('{var}>>histo_nonoise'.format(var=var),'{var}>{xmin} && {var}<{xmax}'.format(var=var,xmin=xmin,xmax=xmax))

    rf_noise = ROOT.TFile.Open("../sim/Reco_Digi_He_6_kev_noise_petrucci.root")
    t_noise = rf_noise.Get('Events')
    histo_noise = ROOT.TH1F('histo_noise','',nbins,xmin,xmax)
    histo_noise.SetMarkerStyle(ms_noise)
    histo_noise.SetLineColor(ROOT.kBlack)
    t_noise.Draw('{var}>>histo_noise'.format(var=var),'{var}>{xmin} && {var}<{xmax}'.format(var=var,xmin=xmin,xmax=xmax))

    rooData_nonoise = ROOT.RooDataHist("histo_nonoise","histo_nonoise",ROOT.RooArgList(work.var("x")),histo_nonoise)
    rooData_noise = ROOT.RooDataHist("histo_noise","histo_noise",ROOT.RooArgList(work.var("x")),histo_noise)    
    
    getattr(work,'import')(rooData_nonoise)
    getattr(work,'import')(rooData_noise)

    frame = x.frame()
    frame.SetTitle('')
    
    # fit noiseless sim
    rooData_nonoise.plotOn(frame,ROOT.RooFit.MarkerStyle(ms_nonoise))
    pdf = work.pdf('cb')
    pdf.fitTo(rooData_nonoise,ROOT.RooFit.Save(),ROOT.RooFit.PrintLevel(-1))
    pdf.plotOn(frame,ROOT.RooFit.LineColor(lc_nonoise))
    rooData_nonoise.plotOn(frame,ROOT.RooFit.MarkerStyle(ms_nonoise))

    frame.GetYaxis().SetTitle("superclusters")
    frame.GetXaxis().SetTitle("E_{SC}/E_{true}")
    frame.GetXaxis().SetTitleOffset(1.2)
    
    m1 = work.var('mean').getVal()
    s1 = work.var('sigma').getVal()

    
    # fit noisy sim
    rooData_noise.plotOn(frame,ROOT.RooFit.MarkerStyle(ms_noise))
    pdf = work.pdf('cb')
    pdf.fitTo(rooData_noise,ROOT.RooFit.Save(),ROOT.RooFit.PrintLevel(-1))
    pdf.plotOn(frame,ROOT.RooFit.LineColor(lc_noise))
    rooData_noise.plotOn(frame,ROOT.RooFit.MarkerStyle(ms_noise))

    m2 = work.var('mean').getVal()
    s2 = work.var('sigma').getVal()
    
    c = getCanvas()
    frame.Draw()

    plots = [histo_nonoise,histo_noise]
    labels = ['MC truth','MC sim']
    styles = ['pe1','pe1']
    legend = doLegend(plots,labels,styles,corner='TL')
    legend.Draw()
    
    tt1 = ROOT.TLatex(0.65,0.8,'#splitline{{m = {mean:.2f}}}{{#sigma/m = {sigma:.3f}}}'.format(mean=m1,sigma=s1/m1))
    printTLatex(tt1,lc_nonoise)

    tt2 = ROOT.TLatex(0.4,0.5,'#splitline{{m = {mean:.2f}}}{{#sigma/m = {sigma:.3f}}}'.format(mean=m2,sigma=s2/m2))
    printTLatex(tt2,lc_noise)

    for ext in ['pdf','png','root']:
        c.SaveAs('eoveretrue.{ext}'.format(ext=ext))
        
    
if __name__ == "__main__":

    parser = optparse.OptionParser(usage='usage: %prog [opts] ', version='%prog 1.0')
    parser.add_option('', '--make'   , type='string'       , default='tworuns' , help='run simple plots (options = tworuns, evsdist, pmtvsz, cluvspmt, cluvsz, multiplicity, hist1d, hist2d, pedrms, eoveretrue)')
    parser.add_option('', '--outdir' , type='string'       , default='./'      , help='output directory with directory structure and plots')
    parser.add_option('', '--var' , type='string'       , default='integral'      , help='variable to plot the histogram')
    parser.add_option('', '--pos' , type='int'       , default=0      , help='position of the iron source')
    parser.add_option('', '--var2' , type='string'       , default='slimness'      , help='variable2 to plot the histogram 2D')
    
    (options, args) = parser.parse_args()

    ## make the output directory first
    os.system('mkdir -p {od}'.format(od=options.outdir))
    
    if options.make in ['all','multiplicity']:
        plotNClusters()

    if options.make in ['all','tworuns']:
        histograms,entries = fillSpectra()
        odir = options.outdir
        os.system('mkdir -p {od}'.format(od=odir))
        drawSpectra(histograms,odir,entries,normEntries=True)
        os.system('cp ../index.php {od}'.format(od=odir))
        drawROC('density',odir)
        
    if options.make in ['all','evsdist']:
        plotEnergyVsDistance(options.outdir)

    if options.make in ['all','pmtvsz']:
        plotPMTEnergyVsPosition(options.outdir)

    if options.make in ['all','cluvspmt']:
        plotCameraPMTCorr(options.outdir)
        
    if options.make in ['all','cluvsz']:
        os.system('mkdir -p {od}'.format(od=options.outdir))
        os.system('cp ../index.php {od}'.format(od=options.outdir))
        plotCameraEnergyVsPosition(options.outdir, options.var)
    
    if options.make in ['all','hist1d']:        
        plotHistFit(options.outdir, options.var, options.pos)
        
    if options.make in ['all','hist2d']:
        plotHist2D(options.outdir, options.var, options.var2, options.pos)

    if options.make in ['all','pedrms']:
        plotPedRMS()

    if options.make in ['all','eoveretrue']:
        plotSimEoverEtrue()
