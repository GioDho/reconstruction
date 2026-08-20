import numpy as np
from waveform import PMTreco
import ROOT


class AutoFillTreeProducer:
    def __init__(self,tree,eventContent):
        self.outTree = tree
        self.eventContent = eventContent

    def createEnvVariables(self):
        self.outTree.branch('Lime_pressure', 'F', title="Lime pressure")
        self.outTree.branch('Atm_pressure', 'F', title="Atmosperic pressure")
        self.outTree.branch('Lime_temperature', 'F', title="Lime temperature")
        self.outTree.branch('Atm_temperature', 'F', title="Atmosheric temperature")
        self.outTree.branch('Humidity', 'F', title="Humidity")
        self.outTree.branch('Oxygen', 'F', title="Oxygen")
        self.outTree.branch('Mixture_Density', 'F', title="Mixture_Density")
        

    def fillEnvVariables(self, dslow):
        env_var = open('modules_config/env_variables.txt','r')
        env_var = eval(env_var.read())
        self.outTree.fillBranch('Lime_pressure', dslow[env_var['lime_pressure']])        
        self.outTree.fillBranch('Atm_pressure', dslow[env_var['atm_pressure']])
        self.outTree.fillBranch('Lime_temperature', dslow[env_var['lime_temperature']])
        self.outTree.fillBranch('Atm_temperature', dslow[env_var['atm_temperature']])
        self.outTree.fillBranch('Humidity', dslow[env_var['humidity']])
        self.outTree.fillBranch('Oxygen', dslow[env_var['oxygen']])
        self.outTree.fillBranch('Mixture_Density', dslow[env_var['mixture_density']])

    ####################################################     PMT    ################################################################################################################################

    """
    # NB:
    # There is a code that allows to create a single TTree with all the waveform information of a single picture clustered together.
    # All the information is rolled down to 1D vectors -> Requires post analysis to disentagle information & specific way of filling branches
    # To be discussed when reconstruction will be rebuild. Contact David for more info.
    """

    ## Each entry in the PMT tree is an individual waveform.
    def createPMTVariables(self,pmt_params):

        # General waveform info 
        ## Identifiable with 'pmt_wf_ID'
        # self.outTree.branch('pmt_nWaveforms',       'I', title  = 'Number of waveforms per event')
        self.outTree.branch('pmt_wf_ID',            'I', title = 'Waveform identifier')             ## ID = {pic}{trigger}{channel}
        self.outTree.branch('pmt_wf_run',           'I', title = 'Waveform run')
        self.outTree.branch('pmt_wf_event',         'I', title = 'Waveform event(picture)')
        self.outTree.branch('pmt_wf_trigger',       'I', title = 'Waveform trigger')
        self.outTree.branch('pmt_wf_channel',       'I', title = 'Waveform channel')
        self.outTree.branch('pmt_wf_insideGE',      'I', title = 'Check if waveform inside Global exposure')
        self.outTree.branch('pmt_wf_sampling',      'I', title = 'Waveform coming from fast or slow digitizer')
        self.outTree.branch('pmt_wf_TTT',           'F', title = 'Waveform/Trigger time of arrival')

        # Waveform physics info 
        self.outTree.branch('pmt_wf_baseline',      'F', title = 'Waveform baseline')
        self.outTree.branch('pmt_wf_RMS',           'F', title = 'Waveform RMS')
        self.outTree.branch('pmt_wf_tot_integral',  'F', title = 'Whole waveform total integrated amplitude')
        self.outTree.branch('pmt_wf_tot_charge',    'F', title = 'Whole waveform total integrated charge')
        self.outTree.branch('pmt_wf_max_ampl',      'F', title = 'Waveform max voltage')
        self.outTree.branch('pmt_wf_nPeaks',        'I', title = 'Waveform number of peaks')

        ## Time Over Threshold
        self.outTree.branch('pmt_TOT_time',             'F', title = "Time over threshold")
        self.outTree.branch('pmt_TOT_area',             'F', title = "Area over threshold")

        ## Full waveforms -- If user requires full raw waveforms to be saved directly in the tree
        #  Identifiable with 'pmt_wf_ID_full'
        if pmt_params['wf_in_tree'] == True:
            self.outTree.branch('pmt_wf_ID_full',       'F', lenVar = 'pmt_full_sized_wf',      title = 'Full waveforms length ID')
            # self.outTree.branch('pmt_fullWaveform_X',   'F', lenVar = 'pmt_full_sized_wf',      title = 'Full waveform for in depth analysis') # x is fixed 
            self.outTree.branch('pmt_fullWaveform_Y',   'F', lenVar = 'pmt_full_sized_wf',      title = 'Full waveform for in depth analysis')

        ## Peak variables -- Leafs
        self.outTree.branch('pmt_peak_Number',      'I', lenVar = 'pmt_wf_nPeaks',  title = 'Peaks numbers')
        self.outTree.branch('pmt_peak_Position',    'F', lenVar = 'pmt_wf_nPeaks',  title = 'Peaks positions')
        self.outTree.branch('pmt_peak_Height',      'F', lenVar = 'pmt_wf_nPeaks',  title = 'Peaks heights')
        self.outTree.branch('pmt_peak_HalfWidth',   'F', lenVar = 'pmt_wf_nPeaks',  title = 'Peaks half widths')
        self.outTree.branch('pmt_peak_FullWidth',   'F', lenVar = 'pmt_wf_nPeaks',  title = 'Peaks full widths')

    def fillPMTVariables(self,wf):

        self.outTree.fillBranch('pmt_wf_ID',                wf.getWaveformID('waveforms')[0])
        self.outTree.fillBranch('pmt_wf_run',               wf.getRun())
        self.outTree.fillBranch('pmt_wf_event',             wf.getEvent())
        self.outTree.fillBranch('pmt_wf_trigger',           wf.getTrigger())
        self.outTree.fillBranch('pmt_wf_channel',           wf.getChannel())
        self.outTree.fillBranch('pmt_wf_insideGE',          wf.getInGE())
        self.outTree.fillBranch('pmt_wf_sampling',          wf.getSampling())
        self.outTree.fillBranch('pmt_wf_TTT',               wf.getTTT())

        self.outTree.fillBranch('pmt_wf_baseline',          wf.getBaseline())
        self.outTree.fillBranch('pmt_wf_RMS',               wf.getRMS())
        self.outTree.fillBranch('pmt_wf_tot_integral',      wf.getTotalIntegral())
        self.outTree.fillBranch('pmt_wf_tot_charge',        wf.voltageToCharge(wf.getTotalIntegral()))
        self.outTree.fillBranch('pmt_wf_max_ampl',          wf.getMaxAmpl())
        self.outTree.fillBranch('pmt_wf_nPeaks',            len(wf.getPeaks()))

        self.outTree.fillBranch('pmt_TOT_time',                 wf.getTOT('time'))
        self.outTree.fillBranch('pmt_TOT_area',                 wf.getTOT('area'))

        if (wf.getWfSaveInfo() == True):
            self.outTree.fillBranch('pmt_wf_ID_full',       [wf_id_p for wf_id_p in wf.getWaveformID('fullWF')])
            # self.outTree.fillBranch('pmt_fullWaveform_X',   [x for x in wf.getFullwaveform('x')])
            self.outTree.fillBranch('pmt_fullWaveform_Y',   [y for y in wf.getFullwaveform('y')])

        self.outTree.fillBranch('pmt_peak_Number',          [pid for pid in wf.getPeakIdentifier()])
        self.outTree.fillBranch('pmt_peak_Position',        [pp for pp in wf.getPeaksPositions()])
        self.outTree.fillBranch('pmt_peak_Height',          [ph for ph in wf.getAmplitudes()])
        self.outTree.fillBranch('pmt_peak_HalfWidth',       [phw for phw in wf.getPeakWidths('half')])
        self.outTree.fillBranch('pmt_peak_FullWidth',       [pfw for pfw in wf.getPeakWidths('full')])


    ######################### Weighted average waveform #########################

    ## The weighted average waveform is used to retrieve the time over threshold. Does not hold value concerning amplitudes
    ##  Useful in the future once we want to save the PHYSICS EVENT variables mixing automatically Camera and PMT reconstructions
    def createPMTVariables_average(self,pmt_params):

        self.outTree.branch('pmt_wf_run',           'I', title = 'Waveform run')
        self.outTree.branch('pmt_wf_event',         'I', title = 'Waveform event/picture')
        self.outTree.branch('pmt_wf_trigger',       'I', title = 'Waveform trigger')
        self.outTree.branch('pmt_wf_channel',       'I', title = 'Waveform channel')            # Fixed to channel 9
        self.outTree.branch('pmt_wf_insideGE',      'I', title = 'Check if waveform inside Global exposure')
        self.outTree.branch('pmt_wf_sampling',      'I', title = 'Waveform coming from fast or slow digitizer')
        self.outTree.branch('pmt_wf_TTT',           'F', title = 'Waveform/Trigger time of arrival')

        self.outTree.branch('pmt_wf_nPeaks',        'I', title = 'Waveform number of peaks')

        self.outTree.branch('pmt_TOT_time',             'F', title = "Time over threshold")
        self.outTree.branch('pmt_TOT_area',             'F', title = "Area over threshold")

        self.outTree.branch('pmt_peak_Number',      'I', lenVar = 'pmt_wf_nPeaks',  title = 'Peaks numbers')
        self.outTree.branch('pmt_peak_Position',    'F', lenVar = 'pmt_wf_nPeaks',  title = 'Peaks positions')
        self.outTree.branch('pmt_peak_Height',      'F', lenVar = 'pmt_wf_nPeaks',  title = 'Peaks heights')
        self.outTree.branch('pmt_peak_HalfWidth',   'F', lenVar = 'pmt_wf_nPeaks',  title = 'Peaks half widths')
        self.outTree.branch('pmt_peak_FullWidth',   'F', lenVar = 'pmt_wf_nPeaks',  title = 'Peaks full widths')

    def fillPMTVariables_average(self,wf):

        self.outTree.fillBranch('pmt_wf_run',               wf.getRun())
        self.outTree.fillBranch('pmt_wf_event',             wf.getEvent())
        self.outTree.fillBranch('pmt_wf_trigger',           wf.getTrigger())
        self.outTree.fillBranch('pmt_wf_channel',           wf.getChannel())                    # Fixed to channel 9
        self.outTree.fillBranch('pmt_wf_insideGE',          wf.getInGE())
        self.outTree.fillBranch('pmt_wf_sampling',          wf.getSampling())
        self.outTree.fillBranch('pmt_wf_TTT',               wf.getTTT())

        self.outTree.fillBranch('pmt_wf_nPeaks',            len(wf.getPeaks()))

        self.outTree.fillBranch('pmt_TOT_time',                 wf.getTOT('time'))
        self.outTree.fillBranch('pmt_TOT_area',                 wf.getTOT('area'))

        self.outTree.fillBranch('pmt_peak_Number',          [pid for pid in wf.getPeakIdentifier()])
        self.outTree.fillBranch('pmt_peak_Position',        [pp for pp in wf.getPeaksPositions()])
        self.outTree.fillBranch('pmt_peak_Height',          [ph for ph in wf.getAmplitudes()])
        self.outTree.fillBranch('pmt_peak_HalfWidth',       [phw for phw in wf.getPeakWidths('half')])
        self.outTree.fillBranch('pmt_peak_FullWidth',       [pfw for pfw in wf.getPeakWidths('full')])



    ########################################################  CAMERA   ############################################################################################################################

    def createCameraVariables(self):
        self.outTree.branch('cmos_integral', 'F', title="integral counts of the full CMOS sensor")
        self.outTree.branch('cmos_mean',     'F', title="average counts of the full CMOS sensor")
        self.outTree.branch('cmos_rms',      'F', title="RMS of the counts of the full CMOS sensor")
        self.outTree.branch('timestamp',      'L', title="Timestamp in UTC of the picture")
        self.outTree.branch('cam',      'I', title="Camera number")

    def createTimeCameraVariables(self):
        self.outTree.branch('t_DBSCAN', 'F', title="DBSCAN time")
        self.outTree.branch('t_variables',     'F', title="Variables time")
        self.outTree.branch('lp_len',     'F', title="# pixel")
        
        self.outTree.branch('t_pedsub',     'F', title="pedestal subtraction")
        self.outTree.branch('t_saturation',     'F', title="saturation correction mode")
        self.outTree.branch('t_zerosup',     'F', title="zero suppression")
        self.outTree.branch('t_xycut',     'F', title="xy acceptance cut")
        self.outTree.branch('t_rebin',     'F', title="rebinning")

        self.outTree.branch('t_medianfilter',     'F', title="median filter")
        self.outTree.branch('t_noisered',     'F', title="noise reductor")

    def createTimePMTVariables(self):
        self.outTree.branch('t_waveforms', 'F', title="waveforms time")
        
    def createClusterVariables(self,name='track'):
        chars = list(name)
        start = chars[0]; rest = chars[1:]
        sizeStr = 'n'+start.upper()+''.join(rest)
        self.outTree.branch('{name}_size'.format(name=name),         'F', lenVar=sizeStr, title="number of pixels of the cluster, without zero-suppression")
        self.outTree.branch('{name}_nhits'.format(name=name),        'F', lenVar=sizeStr, title="number of pixels of the cluster above zero-suppression threshold")
        self.outTree.branch('{name}_nhits_highThr'.format(name=name),'F', lenVar=sizeStr, title="number of pixels of the cluster above 10 ADC counts above pedestal (core of the track)")
        self.outTree.branch('{name}_integral'.format(name=name),     'F', lenVar=sizeStr, title="uncalibrated integral of counts of all the pixels in the cluster")
        self.outTree.branch('{name}_corrintegral'.format(name=name), 'F', lenVar=sizeStr, title="density-corrected integral of the cluster (LEMON-specific calibration)")
        self.outTree.branch('{name}_rms'.format(name=name),          'F', lenVar=sizeStr, title="RMS of counts of all the pixels in the cluster")
        # filled only for the supercluster
        if name=='sc':
            self.outTree.branch('{name}_energy'.format(name=name),  'F', lenVar=sizeStr, title="calibrated energy of the cluster in keV (LEMON-specific calibration)")
            self.outTree.branch('{name}_pathlength'.format(name=name),    'F', lenVar=sizeStr, title="curved length of the cluster (made with skeletonization)")
            if self.eventContent["scfullinfo"] == True:
                self.outTree.branch('{name}_redpixIdx'.format(name=name),   'F',  lenVar=sizeStr, title="index of the first pixel in the reduced pixel (redpix) collection belonging to the cluster")
                self.outTree.branch('redpix_ix',        'I', lenVar='nRedpix', title="x coordinate of the pixel") 
                self.outTree.branch('redpix_iy',        'I', lenVar='nRedpix', title="y coordinate of the pixel") 
                self.outTree.branch('redpix_iz',        'F', lenVar='nRedpix', title="number of counts of the pixel (after pedestal subtraction)") 
        self.outTree.branch('{name}_theta'.format(name=name),        'F', lenVar=sizeStr, title="polar angle inclination of the major-axis of the cluster")
        self.outTree.branch('{name}_length'.format(name=name),       'F', lenVar=sizeStr, title="length of the major axis of the cluster")
        self.outTree.branch('{name}_width'.format(name=name),        'F', lenVar=sizeStr, title="length of the minor axis of the cluster")
        self.outTree.branch('{name}_longrms'.format(name=name),      'F', lenVar=sizeStr, title="truncated RMS of the cluster along the major axis")
        self.outTree.branch('{name}_latrms'.format(name=name),       'F', lenVar=sizeStr, title="truncated RMS of the cluster along the minor axis")
        self.outTree.branch('{name}_lfullrms'.format(name=name),     'F', lenVar=sizeStr, title="full RMS of the cluster along the major axis")
        self.outTree.branch('{name}_tfullrms'.format(name=name),     'F', lenVar=sizeStr, title="full RMS of the cluster along the minor axis")
        self.outTree.branch('{name}_lp0amplitude'.format(name=name), 'F', lenVar=sizeStr, title="amplitude of the main peak of the longitudinal cluster profile")
        self.outTree.branch('{name}_lp0prominence'.format(name=name),'F', lenVar=sizeStr, title="prominence of the main peak wrt the local baseline along the longitudinal cluster profile")
        self.outTree.branch('{name}_lp0fwhm'.format(name=name),      'F', lenVar=sizeStr, title="full width at half-maximum of the main peak of the longitudinal cluster profile")
        self.outTree.branch('{name}_lp0mean'.format(name=name),      'F', lenVar=sizeStr, title="mean position wrt the start of the cluster of the main peak of the longitudinal cluster profile")
        self.outTree.branch('{name}_tp0fwhm'.format(name=name),      'F', lenVar=sizeStr, title="full width at half-maximum of the main peak of the transverse cluster profile")
        self.outTree.branch('{name}_xmean'.format(name=name),        'F', lenVar=sizeStr, title="x position of the cluster energy baricenter")
        self.outTree.branch('{name}_ymean'.format(name=name),        'F', lenVar=sizeStr, title="y position of the cluster energy baricenter")
        self.outTree.branch('{name}_xmax'.format(name=name),         'F', lenVar=sizeStr, title="x position of the rightmost pixel of the cluster")
        self.outTree.branch('{name}_xmin'.format(name=name),         'F', lenVar=sizeStr, title="x position of the leftmost pixel of the cluster")
        self.outTree.branch('{name}_ymax'.format(name=name),         'F', lenVar=sizeStr, title="y position of the topmost pixel of the cluster")
        self.outTree.branch('{name}_ymin'.format(name=name),         'F', lenVar=sizeStr, title="y position of the bottommost pixel of the cluster")
        self.outTree.branch('{name}_pearson'.format(name=name),      'F', lenVar=sizeStr, title="Pearson coefficient of the cluster")
        self.outTree.branch('{name}_tgaussamp'.format(name=name),    'F', lenVar=sizeStr, title="amplitude of the Gaussian transverse profile")
        self.outTree.branch('{name}_tgaussmean'.format(name=name),   'F', lenVar=sizeStr, title="mean position of the Gaussian transverse profile")
        self.outTree.branch('{name}_tgausssigma'.format(name=name),  'F', lenVar=sizeStr, title="standard deviation of the Gaussian transverse profile")
        self.outTree.branch('{name}_tchi2'.format(name=name),        'F', lenVar=sizeStr, title="chi-squared of the Gaussian fit to the transverse profile")
        self.outTree.branch('{name}_tstatus'.format(name=name),      'F', lenVar=sizeStr, title="status of the Gaussian fit to the transverse profile")
        self.outTree.branch('{name}_lgaussamp'.format(name=name),    'F', lenVar=sizeStr, title="amplitude of the Gaussian longitudinal profile")
        self.outTree.branch('{name}_lgaussmean'.format(name=name),   'F', lenVar=sizeStr, title="mean position of the Gaussian longitudinal profile")
        self.outTree.branch('{name}_lgausssigma'.format(name=name),  'F', lenVar=sizeStr, title="standard deviation of the Gaussian longitudinal profile")
        self.outTree.branch('{name}_lchi2'.format(name=name),        'F', lenVar=sizeStr, title="chi-squared of the Gaussian fit to the longitudinal profile")
        self.outTree.branch('{name}_lstatus'.format(name=name),      'F', lenVar=sizeStr, title="status of the Gaussian fit to the longitudinal profile")

    def fillCameraVariables(self,pic,timestamp,camera_n):
        self.outTree.fillBranch('cmos_integral',np.sum(pic))
        self.outTree.fillBranch('cmos_mean',np.mean(pic))
        self.outTree.fillBranch('cmos_rms',np.std(pic))
        self.outTree.fillBranch('timestamp',timestamp)
        self.outTree.fillBranch('cam',camera_n)

    def fillTimeCameraVariables(self, t_variables, t_DBSCAN, lp, t_pedsub, t_saturation, t_zerosup, t_xycut, t_rebin, t_medianfilter, t_noisered):
        self.outTree.fillBranch('t_DBSCAN', t_DBSCAN)
        self.outTree.fillBranch('t_variables',t_variables)
        self.outTree.fillBranch('lp_len',lp)

        self.outTree.fillBranch('t_pedsub', t_pedsub)
        self.outTree.fillBranch('t_saturation', t_saturation)
        self.outTree.fillBranch('t_zerosup', t_zerosup)
        self.outTree.fillBranch('t_xycut', t_xycut)
        self.outTree.fillBranch('t_rebin', t_rebin)

        self.outTree.fillBranch('t_medianfilter', t_medianfilter)
        self.outTree.fillBranch('t_noisered', t_noisered) 

    def fillTimePMTVariables(self, t_waveforms):
        self.outTree.fillBranch('t_waveforms', t_waveforms)

    def fillClusterVariables(self,clusters,name='track'):
        chars = list(name)
        start = chars[0]; rest = chars[1:]
        sizeStr = 'n'+start.upper()+''.join(rest)
        self.outTree.fillBranch('{name}_size'.format(name=name),     [cl.size() for cl in clusters])
        self.outTree.fillBranch('{name}_nhits'.format(name=name),    [cl.sizeActive() for cl in clusters])
        self.outTree.fillBranch('{name}_nhits_highThr'.format(name=name),    [cl.sizeActive_highThr() for cl in clusters])
        self.outTree.fillBranch('{name}_integral'.format(name=name), [cl.integral() for cl in clusters])
        self.outTree.fillBranch('{name}_corrintegral'.format(name=name), [cl.corr_integral() for cl in clusters])
        self.outTree.fillBranch('{name}_rms'.format(name=name),      [cl.rms() for cl in clusters])
        # filled only for the supercluster
        if name=='sc':
            self.outTree.fillBranch('{name}_energy'.format(name=name), [cl.calibratedEnergy for cl in clusters])
            self.outTree.fillBranch('{name}_pathlength'.format(name=name),    [cl.pathlength for cl in clusters])
            if self.eventContent["scfullinfo"] == True:
                selection = self.eventContent["scpixels_sel"]
                redPixIdxs = []
                ix = []; iy = []; iz = []
                for cl in clusters:
                    if cl.shapes['long_width'] < float(selection["max_len"]) and cl.integral() > float(selection["min_integral"]):
                        redPixIdxs.append(len(ix))
                        ix = ix + [round(cl.xallpixelcoord[i]) for i in range(cl.nallintpixels)]
                        iy = iy + [round(cl.yallpixelcoord[i]) for i in range(cl.nallintpixels)]
                        iz = iz + [round(cl.zallpixel[i]*10)/10. for i in range(cl.nallintpixels)]
                    else:
                        redPixIdxs.append(-1)
                self.outTree.fillBranch('redpix_ix', ix)
                self.outTree.fillBranch('redpix_iy', iy)
                self.outTree.fillBranch('redpix_iz', iz)
                self.outTree.fillBranch('{name}_redpixIdx'.format(name=name),   redPixIdxs)
        self.outTree.fillBranch('{name}_theta'.format(name=name),    [cl.shapes['theta'] for cl in clusters])
        self.outTree.fillBranch('{name}_length'.format(name=name),   [cl.shapes['long_width'] for cl in clusters])
        self.outTree.fillBranch('{name}_width'.format(name=name),    [cl.shapes['lat_width'] for cl in clusters])
        self.outTree.fillBranch('{name}_longrms'.format(name=name),  [cl.shapes['longrms'] for cl in clusters])
        self.outTree.fillBranch('{name}_latrms'.format(name=name),   [cl.shapes['latrms'] for cl in clusters])
        self.outTree.fillBranch('{name}_lfullrms'.format(name=name), [cl.shapes['long_fullrms'] for cl in clusters])
        self.outTree.fillBranch('{name}_tfullrms'.format(name=name), [cl.shapes['lat_fullrms'] for cl in clusters])

        self.outTree.fillBranch('{name}_lp0amplitude'.format(name=name), [cl.shapes['long_p0amplitude'] for cl in clusters])
        self.outTree.fillBranch('{name}_lp0prominence'.format(name=name), [cl.shapes['long_p0prominence'] for cl in clusters])
        self.outTree.fillBranch('{name}_lp0fwhm'.format(name=name),   [cl.shapes['long_p0fwhm'] for cl in clusters])
        self.outTree.fillBranch('{name}_lp0mean'.format(name=name),   [cl.shapes['long_p0mean'] for cl in clusters])
        self.outTree.fillBranch('{name}_tp0fwhm'.format(name=name),   [cl.shapes['lat_p0fwhm'] for cl in clusters])

        self.outTree.fillBranch('{name}_xmean'.format(name=name),     [cl.shapes['xmean'] for cl in clusters])
        self.outTree.fillBranch('{name}_ymean'.format(name=name),     [cl.shapes['ymean'] for cl in clusters])
        self.outTree.fillBranch('{name}_xmax'.format(name=name),      [cl.shapes['xmax'] for cl in clusters])
        self.outTree.fillBranch('{name}_xmin'.format(name=name),      [cl.shapes['xmin'] for cl in clusters])
        self.outTree.fillBranch('{name}_ymax'.format(name=name),      [cl.shapes['ymax'] for cl in clusters])
        self.outTree.fillBranch('{name}_ymin'.format(name=name),      [cl.shapes['ymin'] for cl in clusters])
        self.outTree.fillBranch('{name}_pearson'.format(name=name),   [cl.getPearson() for cl in clusters])
        self.outTree.fillBranch('{name}_tgaussamp'.format(name=name),   [cl.shapes['tgaussamp'] for cl in clusters])
        self.outTree.fillBranch('{name}_tgaussmean'.format(name=name),  [cl.shapes['tgaussmean'] for cl in clusters])
        self.outTree.fillBranch('{name}_tgausssigma'.format(name=name), [cl.shapes['tgausssigma'] for cl in clusters])
        self.outTree.fillBranch('{name}_tchi2'.format(name=name),       [cl.shapes['tchi2'] for cl in clusters]) 
        self.outTree.fillBranch('{name}_tstatus'.format(name=name),     [cl.shapes['tstatus'] for cl in clusters])
        self.outTree.fillBranch('{name}_lgaussamp'.format(name=name),   [cl.shapes['lgaussamp'] for cl in clusters])
        self.outTree.fillBranch('{name}_lgaussmean'.format(name=name),  [cl.shapes['lgaussmean'] for cl in clusters])
        self.outTree.fillBranch('{name}_lgausssigma'.format(name=name), [cl.shapes['lgausssigma'] for cl in clusters])
        self.outTree.fillBranch('{name}_lchi2'.format(name=name),       [cl.shapes['lchi2'] for cl in clusters]) 
        self.outTree.fillBranch('{name}_lstatus'.format(name=name),     [cl.shapes['lstatus'] for cl in clusters])
