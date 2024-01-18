#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from os.path import join as opj
from glob import glob as gg
import os, time
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
mpl.use('Agg')
from mne import (sys_info, read_annotations, events_from_annotations, Epochs, add_reference_channels)
from mne.channels import make_standard_montage
from mne.io import read_raw_brainvision as read_bv
from mne.io import RawArray
from mne.preprocessing import ICA
print(sys_info())
(lc1, hc1) = (0.1, None)
(lc, hc) = (1, 100) # (Hz), low and high cutoffs of filters before ICA
RootPath = '~/Experiments/SPF'
EegPath = opj(RootPath,'raw','eeg')
BhvPath = opj(RootPath,'raw','bhv')
OutPath = opj(RootPath,'output')
OutPath1 = opj(OutPath,'step_1')
if not os.path.exists(OutPath1): os.makedirs(OutPath1)
OutPathIca = opj(OutPath,'ICA')
if not os.path.exists(OutPathIca): os.makedirs(OutPathIca)

## load standard layout
ten_twenty_montage = make_standard_montage('standard_1020')
fig = ten_twenty_montage.plot(kind='topomap', show_names=True)
# save the layout image
fig.savefig(opj(OutPath,'std1020_chloc.png'))
# subject lists
SubjFiles = gg(opj(EegPath,'SPF*.vhdr'))
SubjFiles.sort()
## 
SubInc = []
# cells containing 0.1 high-pass filtered raw and 1-100 band-pass filtered raw
rwnames = ['%s-%sHzfiltered'%(str(lc1), str(hc1)), '%s-%sHzfiltered'%(str(lc), str(hc))]
## subject-level loop
for subj in SubjFiles:
    # subject number
    SubNum = subj.split('/')[-1].split('.')[0]
    if os.path.exists(opj(OutPathIca, SubNum + '-ica.fif')): continue
    if SubNum == 'SPF129': 
        subj = opj(subj, 'SPF126.vhdr')
    elif SubNum == 'SPF130':
        subj = opj(subj, '130.vhdr')
    # load behavioral data as pandas dataframe
    bhvfile = gg(opj(BhvPath, SubNum + '*_behav.txt'))[0]
    bhv = pd.read_csv(bhvfile, delimiter = '\t')
    bhv = bhv.query("run != 0")
    # read raw EEG
    raw = read_bv(subj,
                  eog = ['FT9', 'FT10'], 
                  preload = True)
    print(SubNum, raw.info)
    # reverse amplifiers for SPF101, SPF102, SPF103 as their wrong device settings
    if SubNum in ['SPF101','SPF102','SPF103']:
        data1 = raw.get_data()
        data2 = data1.copy()
        data2[0:32,:] = data1[32:,:]
        data2[32:,:] = data1[0:32,:]
        raw = RawArray(data=data2, info=raw.info)
    # recover online reference electrode FCz
    raw = add_reference_channels(raw, 'FCz')
    # set 10-20 montage
    raw.set_montage(ten_twenty_montage)
    # read marker file
    markers = read_annotations(subj[:-5] + '.vmrk', sfreq = raw.info['sfreq'])
    raw.set_annotations(markers)
    print(raw.info['meas_date'] == raw.annotations.orig_time)
    # rereference
    raw = raw.set_eeg_reference(ref_channels='average', projection=False, ch_type='eeg')
    # define events
    events_mapping = {'Stimulus/S211':211, 'Stimulus/S221':221,
                      'Stimulus/S231':231, 'Stimulus/S241':241}
    events_from_annot, event_dict = events_from_annotations(raw, event_id = events_mapping)
    event_dict1 = {'Ans1':211, 'Ans2':221, 'Prm':231, 'Req':241}
    #### epoch ####
    for rwname in rwnames:
        if float(rwname.split('-')[0]) == lc1:
            rw = raw.copy().filter(l_freq = lc1, h_freq = hc1)
            print('Processing %s'%rwname)
        else:
            rw = raw.copy().filter(l_freq = lc, h_freq = hc)
            print('Processing %s'%rwname)
        # epochs
        epochs = Epochs(rw, 
                        events = events_from_annot, 
                        event_id = event_dict1, 
                        tmin = -13.9, tmax = 0.8*4+0.4*3+1,
                        reject_tmin = 0.,
                        reject_tmax = 0.8*4+0.4*3,
                        baseline = None,
                        preload = True)
        del rw
        epochs.metadata = bhv
        # save epoched data
        epochs.save(opj(OutPath1, SubNum + '_%s_caref-epo.fif'%rwname), overwrite = True)
        #### ICA ####
        if rwname == rwnames[1]:
            ica = ICA(method = 'infomax', fit_params = dict(extended = True), random_state = int(time.time()))
            ica.fit(epochs, picks = 'eeg')
            # save ICA solution
            ica.save(opj(OutPathIca, SubNum + '-ica.fif'))
            del ica
        del epochs
    del raw
