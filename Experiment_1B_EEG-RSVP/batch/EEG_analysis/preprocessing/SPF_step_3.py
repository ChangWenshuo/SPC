#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from os.path import join as opj
from glob import glob as gg
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.pyplot import close
mpl.use('Agg')
from mne import read_epochs, sys_info
print(sys_info())

def reject_bads(epc, t1, t2, threshold=dict(eeg=120e-6)):
    epc.set_channel_types({cc:'ecg' for cc in epc.ch_names\
                             if ('Fp' in cc) or ('AF' in cc) or (cc in ['T7','T8','TP9','TP10'])},\
                        verbose=None)
    if not (t1==0 and t2==0): (epc.reject_tmin, epc.reject_tmax) = (t1, t2)
    epc = epc.drop_bad(reject=threshold)
    epc.set_channel_types({cc:'eeg' for cc in epc.ch_names\
                             if ('Fp' in cc) or ('AF' in cc) or (cc in ['T7','T8','TP9','TP10'])},\
                        verbose=None)
    return epc

ExpNum = 'SPF'
RootPath = f'~/Experiments/{ExpNum}'
pars = np.loadtxt(opj(RootPath, 'batch/prep_2309/step_3.txt'), dtype = str)
hc_rej = 30 # (Hz) highpass filter for artifact rejection
thr = 100
bsl = -0.1
rej_thr = thr*10**-6
BhvPath = opj(RootPath,'raw','bhv')
OutPath = opj(RootPath,'output')
EegPath2 = opj(OutPath,'step_2')
OutPath3 = opj(OutPath,'step_3_%slowpass_thr%i'%(str(hc_rej), thr))
if not os.path.exists(OutPath3): os.makedirs(OutPath3)
Crts = {'crt34':(0.8*2+0.4*2, 0.8*4+0.4*3)}

(lc1, hc1) = (0.1, None)
(lc, hc) = (1, 100) # (Hz), cutoff frequency of filter for ICA
epnames = ['%s-%sHzfiltered'%(str(lc1), str(hc1)), '%s-%sHzfiltered'%(str(lc), str(hc))]
# subject lists
SubjFiles = gg(opj(EegPath2, '%s*%s*_recon-epo.fif'%(ExpNum, epnames[0])))
SubjFiles.sort()

data_sur = dict(SubNum=[],Ans1=[],Prm=[],Ans2=[],Req=[])
## subject-level loop
for subj in SubjFiles:
    # subject number
    SubNum = subj.split('/')[-1].split('_')[0]
    # read 1-100 Hz bandpass filtered data
    epochs2 = read_epochs(subj.split(epnames[0])[0] + '%s_caref_recon-epo.fif'%epnames[1], preload = True)
    epochs2 = epochs2.pick(picks='eeg')
    # low-pass filter
    epochs2 = epochs2.filter(l_freq = None, h_freq = hc_rej)
    #　reject bad epochs
    ## time window for artifact rejection, 
    ## lower bound is the start point of the baseline
    ## upper bound is the maxima of gaze durations
    (t1, t2) = (Crts['crt34'][0] + bsl, Crts['crt34'][1])
    # peak-to-peak (PTP) artifact rejection
    epochs2 = reject_bads(epochs2, t1, t2, threshold = dict(eeg = rej_thr))
    include_index = epochs2.metadata.index.copy()
    del epochs2
    # read the 0.1 Hz highpass filtered epoch data
    epochs = read_epochs(subj, preload = True)
    # load behavioral data as pandas dataframe
    bhvfile = gg(opj(BhvPath, SubNum + '*_behav.txt'))[0]
    bhv = pd.read_csv(bhvfile, delimiter = '\t')
    bhv = bhv.query("run != 0")
    epochs.metadata = bhv
    epochs = epochs.pick(picks = 'eeg')
    epochs.save(opj(OutPath3, subj.split('/')[-1].split('-epo')[0] + '_norej-epo.fif'), overwrite = True)
    # the rejected epoch indices
    bad_indices = list(set(epochs.metadata.index) - set(include_index))
    print(f"Rejecting {len(bad_indices)} epochs: {bad_indices}")
    # rection for the unfiltered data
    epochs = epochs.drop(bad_indices, reason = 'PTP_for_30Hzlowpass_data')
    # how many epochs survived
    data_sur['SubNum'].extend([SubNum])
    for ev in epochs.event_id.keys(): data_sur[ev].extend([len(epochs[ev])])
    ## save summary table of survived epochs
    data_sur1 = pd.DataFrame(data_sur)
    data_sur1.to_csv(opj(OutPath3, 'data_survived.txt'), sep = '\t', header = True, index = False)
    # save artifact-rejected epoch data
    epochs.save(opj(OutPath3, subj.split('/')[-1].split('-epo')[0] + '_rej-epo.fif'), overwrite = True)
    del epochs
    