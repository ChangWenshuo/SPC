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
import mne
from mne import (sys_info, read_annotations, events_from_annotations, Epochs)
from mne.channels import make_standard_montage
from mne.io import read_raw_eeglab
from mne.channels import read_layout
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs, corrmap)
print(sys_info())
(lc1, hc1) = (0.1, None)
(lc, hc) = (1, 100) # (Hz), low/high-cutoff of band-pass filter before ICA
RootPath = '~/Experiments/SPE'
EegPath = opj(RootPath, 'raw', 'eyeeeg')
BhvPath = opj(RootPath, 'raw', 'bhv')
OutPath = opj(RootPath, 'output')
OutPath1 = opj(OutPath, 'step_1')
if not os.path.exists(OutPath1): os.makedirs(OutPath1)
OutPathIca = opj(OutPath, 'ICA')
if not os.path.exists(OutPathIca): os.makedirs(OutPathIca)
## load standard layout
ten_twenty_montage = make_standard_montage('standard_1020')
fig = ten_twenty_montage.plot(kind = 'topomap', show_names = True)
# save the layout image
fig.savefig(opj(OutPath, 'std1020_chloc.png'))
# subject lists
SubjFiles = gg(opj(EegPath, 'SPE*.set'))
SubjFiles.sort()
# cells containing unfiltered raw and filtered raw
rwnames = ['%s-%sHzfiltered'%(str(lc1), str(hc1)), '%s-%sHzfiltered'%(str(lc), str(hc))]
## subject-level loop
for subj in SubjFiles:
    # subject number
    SubNum = subj.split('/')[-1].split('_')[0]
    if os.path.exists(opj(OutPathIca, SubNum + '-ica.fif')): continue
    # load behavioral data as pandas dataframe
    bhvfile = gg(opj(BhvPath, SubNum + '*_behav.txt'))[0]
    bhv = pd.read_csv(bhvfile, delimiter = '\t')
    bhv = bhv.query("run != 0")
    bhvfile2 = opj(RootPath, 'raw', 'epccheck', SubNum + '_epccheck.txt')
    bhv2 = pd.read_csv(bhvfile2, delimiter = '\t')
    # read raw EEG
    raw = read_raw_eeglab(subj,
                  eog = ['FT9', 'FT10'], 
                  preload = True)
    print(SubNum, raw.info)
    raw.drop_channels(['TIME', 'R-GAZE-X', 'R-GAZE-Y', 'R-AREA'])
    # recover online reference electrode FCz
    raw = mne.add_reference_channels(raw, 'FCz')
    # set 10-20 montage
    raw.set_montage(ten_twenty_montage)
    # read marker file
    markers = read_annotations(subj, sfreq = raw.info['sfreq'])
    raw.set_annotations(markers)
    # check marker file date
    print(raw.info['meas_date'] == raw.annotations.orig_time)
    # define events
    events_mapping = {'S213':213, 'S223':223, 'S233':233, 'S243':243}
    events_from_annot, event_dict = events_from_annotations(raw, event_id = events_mapping)
    event_dict1 = {'Ans1':213, 'Ans2':223, 'Prm':233, 'Req':243}
    # re-reference
    raw = raw.set_eeg_reference(ref_channels='average', projection=False, ch_type='eeg')
    #### epoch ####
    for rwname in rwnames:
        ## highpass filter with low cutoff of 1 Hz for ICA
        if float(rwname.split('-')[0]) == lc1:
            rw = raw.copy().filter(l_freq = lc1, h_freq = hc1)
            print('Processing %s'%rwname)
        else:
            rw = raw.copy().filter(l_freq = lc, h_freq = hc)
            print('Processing %s'%rwname)
        # epoch
        epochs = Epochs(rw, 
                        events = events_from_annot, 
                        event_id = event_dict1, 
                        tmin = -1.5, tmax = 2.5,
                        baseline = None,
                        preload = True)
        del rw
        # down sampling data if higher than 500 Hz
        if epochs.info['sfreq'] > 500.: epochs.resample(500.)
        # remove trials without a fixation at the predicate region
        epochs.metadata = bhv.iloc[list(bhv2.crt34 != 0)]
        # save epoched data
        epochs.save(opj(OutPath1,SubNum+'_%s_caref-epo.fif'%rwname), overwrite=True)
        #### ICA ####
        if rwname == rwnames[1]:
            ica = ICA(method = 'infomax', fit_params = dict(extended = True), random_state = int(time.time()))
            ica.fit(epochs, picks = 'eeg')
            # save ICA solution
            ica.save(opj(OutPathIca, SubNum + '-ica.fif'))
            del ica
        del epochs
    del raw
