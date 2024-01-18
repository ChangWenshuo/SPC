#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from os.path import join as opj
from glob import glob as gg
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
mpl.use('Agg')
from mne import read_epochs, sys_info
from mne.preprocessing import read_ica
from mne_icalabel import label_components
print(sys_info())
## EOG and ECG, and obvious channel noise
SubNames = {
            'SPE114':[0,1,7], # IC7 is ECG
            'SPE115':[0,1], # IC4 is not a typical ECG
            'SPE117':[0,1,6], # IC6 is ECG
            'SPE123':[0,1], # IC36 is not a typical ECG
            'SPE124':[0,1], # IC1 is HEOG
            'SPE127':[0,1,4], # IC10 is not a typical EOG
            'SPE129':[0,1,17], # IC17 is ECG
            'SPE135':[0,1,2,28], # IC2 is HEOG
            'SPE137':[0,1], # IC2 is not a typical EOG
            'SPE141':[0,1,3] # IC1 is VEOG, IC3 is HEOG, IC49 is not a typical ECG
            }

ExpNum = 'SPE'
RootPath = f'/gpfs/share/home/1701110673/Experiments/{ExpNum}'
BhvPath = opj(RootPath,'raw','bhv')
OutPath = opj(RootPath,'output')
EegPath1 = opj(OutPath,'step_1')
IcaPath = opj(OutPath,'ICA')
OutPath2 = opj(OutPath,'step_2')
OutPathPlt = opj(OutPath2, 'plots')
# subject lists
SubjFiles = gg(opj(EegPath1,'%s*_0.1-NoneHzfiltered_caref-epo.fif'%ExpNum))
SubjFiles.sort()
(lc1, hc1) = (0.1, None)
(lc, hc) = (1, 100) # (Hz), cutoff frequency of filter for ICA
epnames = ['%s-%sHzfiltered'%(str(lc1), str(hc1)), '%s-%sHzfiltered'%(str(lc), str(hc))]
## subject-level loop
for subj in SubjFiles:
    # subject number
    SubNum = subj.split('/')[-1].split('_')[0]
    # output directory
    OutPathSub = opj(OutPathPlt, SubNum)
    outtxt = opj(OutPathSub, '%s_ica_exclude_sup.txt'%SubNum)
    if os.path.exists(outtxt): os.system('rm %s'%outtxt)
    if (not SubNum in SubNames.keys()) or (SubNames[SubNum]==[]): continue
    # read 1-100 Hz bandpass filtered data
    epochs2 = read_epochs(gg(subj.split(epnames[0])[0] + '*%s_*-epo.fif'%epnames[1])[0])
    # read ica data
    ica = read_ica(opj(IcaPath,SubNum+'-ica.fif'))
    # icalabels
    ic_labels = label_components(epochs2, ica, method = "iclabel")
    ic_labels = pd.DataFrame(ic_labels)
    # predefined ICs to exclude
    ica.exclude = SubNames[SubNum]
    # save excluded IC list
    list_exclude = pd.DataFrame({'index': ica.exclude, 
                                'label': ic_labels.labels[ica.exclude],
                                'prob': ic_labels.y_pred_proba[ica.exclude]
                                })
    list_exclude.to_csv(outtxt, sep = '\t', index = False)
    del list_exclude, ic_labels
    # ica.apply() changes the Raw object in-place
    ica2 = ica.copy()
    ica2.apply(epochs2)
    ## save the recontructed data
    epochs2.save(opj(OutPath2, subj.split('/')[-1].split(epnames[0])[0] + '%s_recon-epo.fif'%epnames[1]), overwrite = True)
    del epochs2, ica2
    # read 0.1 Hz highpass filtered data
    epochs_recon = read_epochs(subj)
    # plot epochs to compare the reconstructed data with the original data
    fig = epochs_recon.plot(n_epochs = 20, n_channels = 62, show_scrollbars = False)
    fig.savefig(opj(OutPathSub, SubNum+'_epochs_orgin_sup.png'))
    # ica.apply() changes the Raw object in-place
    ica.apply(epochs_recon)
    ## save the recontructed data
    outepc = opj(OutPath2, subj.split('/')[-1].split('-epo')[0] + '_recon-epo.fif')
    epochs_recon.save(outepc, overwrite=True)
    # plot epochs to compare the reconstructed data with the original data
    fig = epochs_recon.plot(n_epochs = 20, n_channels = 62, show_scrollbars = False)
    fig.savefig(opj(OutPathSub, SubNum+'_epochs_recon_sup.png'))
    plt.close('all')
    del epochs_recon, ica, fig
