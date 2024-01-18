#!/usr/bin/env python3
# -*- coding: utf-8 -*
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
eog_thr = 0.2
eye_thr = 0.5
ecg_thr = 0.9
ExpNum = 'SPE'
RootPath = f'~/Experiments/{ExpNum}'
BhvPath = opj(RootPath,'raw','bhv')
OutPath = opj(RootPath,'output')
EegPath1 = opj(OutPath,'step_1')
IcaPath = opj(OutPath,'ICA')
OutPath2 = opj(OutPath,'step_2')
OutPathPlt = opj(OutPath2, 'plots')
if not os.path.exists(OutPath2): os.makedirs(OutPath2)
(lc1, hc1) = (0.1, None)
(lc, hc) = (1, 100) # (Hz), cutoff frequency of filter for ICA
epnames = ['%s-%sHzfiltered'%(str(lc1), str(hc1)), '%s-%sHzfiltered'%(str(lc), str(hc))]
# subject lists
SubjFiles = gg(opj(EegPath1, '%s*%s_*-epo.fif'%(ExpNum, epnames[0])))
SubjFiles.sort()
## subject-level loop
for subj in SubjFiles:
    # subject number
    SubNum = subj.split('/')[-1].split('_')[0]
    # read ica data
    ica = read_ica(opj(IcaPath, SubNum + '-ica.fif'))
    epochs2 = read_epochs(gg(subj.split(epnames[0])[0] + '*%s_*-epo.fif'%epnames[1])[0])
    # output directory
    OutPathSub = opj(OutPathPlt, SubNum)
    OutPathLab = opj(OutPathSub, 'icalabels')
    if not os.path.exists(OutPathLab): os.makedirs(OutPathLab)
    # plot ICs
    if not os.path.exists(opj(OutPathSub, SubNum + '_ICAsol0.png')):
        fig = ica.plot_components()
        for i in range(len(fig)): fig[i].savefig(opj(OutPathSub, SubNum + '_ICAsol' + str(i) + '.png'))
        plt.close('all')
    # icalabels
    ic_labels = label_components(epochs2, ica, method = "iclabel")
    ic_labels = pd.DataFrame(ic_labels)
    elab_indices = ic_labels[(ic_labels.labels == 'eye blink')  & (ic_labels.y_pred_proba >= eye_thr)].index.tolist()
    hlab_indices = ic_labels[(ic_labels.labels == 'heart beat') & (ic_labels.y_pred_proba >= ecg_thr)].index.tolist()
    if not os.path.exists(opj(OutPathLab, SubNum+'_ic0.png')):
        fig = ica.plot_properties(epochs2, picks=range(ica.n_components_))
        for i in range(len(fig)):
            fig[i].suptitle('%s - %.2f'%(ic_labels['labels'][i], ic_labels['y_pred_proba'][i]), x = 0.2, y = 0.98)
            fig[i].savefig(opj(OutPathLab, SubNum+'_ic'+str(i)+'.png'))
        plt.close('all')
    # find which ICs match the EOG pattern
    eog_ind, eog_scores = ica.find_bads_eog(epochs2, threshold = eog_thr, measure = 'correlation')
    eog_ind.sort()
    # save EOG list
    list_ics = pd.DataFrame({'index': eog_ind, 
                                 'label': ic_labels.labels[eog_ind],
                                 'prob': ic_labels.y_pred_proba[eog_ind]
                                 })
    list_ics.to_csv(opj(OutPathSub, '%s_eog.txt'%SubNum), sep = '\t', index = False)
    # barplot of ICA component "EOG match" scores
    fig = ica.plot_scores(eog_scores, exclude = eog_ind)
    fig.savefig(opj(OutPathSub, SubNum + '_eogscore.png'))
    plt.close('all')
    # find which ICs match the EMG pattern
    muscle_ind, muscle_scores = ica.find_bads_muscle(epochs2)
    muscle_ind.sort()
    # save muscle artifact list
    list_ics = pd.DataFrame({'index': muscle_ind, 
                             'label': ic_labels.labels[muscle_ind],
                             'prob': ic_labels.y_pred_proba[muscle_ind]
                            })
    list_ics.to_csv(opj(OutPathSub, '%s_muscle.txt'%SubNum), sep = '\t', index = False)
    # barplot of ICA component "EMG match" scores
    fig = ica.plot_scores(muscle_scores, exclude=muscle_ind)
    fig.savefig(opj(OutPathSub, SubNum + '_emgscore.png'))
    plt.close('all')
    ##### ICs to exclude
    ica.exclude = []
    ica.exclude = list((set(elab_indices) & set(eog_ind)) | set(hlab_indices)) # | (set(mlab_indices) & set(muscle_ind))
    ica.exclude.sort()
    # save excluded IC list
    list_exclude = pd.DataFrame({'index': ica.exclude, 
                                 'label': ic_labels.labels[ica.exclude],
                                 'prob': ic_labels.y_pred_proba[ica.exclude]
                                 })
    list_exclude.to_csv(opj(OutPathSub, '%s_ica_exclude.txt'%SubNum), sep = '\t', index = False)
    del list_exclude, ic_labels
    # ica.apply() changes the 1-100 Hz bandpass filtered data in-place
    ica2 = ica.copy()
    ica2.apply(epochs2)
    ## save the recontructed data
    epochs2.save(opj(OutPath2, subj.split('/')[-1].split(epnames[0])[0] + '%s_caref_recon-epo.fif'%epnames[1]), overwrite = True)
    del epochs2, ica2
    # read 0.1 Hz highpass filtered data
    epochs_recon = read_epochs(subj)
    # plot epochs to compare the reconstructed data with the original data
    fig = epochs_recon.plot(n_epochs = 20, n_channels = 62, show_scrollbars = False)
    fig.savefig(opj(OutPathSub, SubNum + '_epochs_orgin.png'))
    # ica.apply() changes the 0.1 Hz highpass filtered data in-place
    ica.apply(epochs_recon)
    print(f"Excluding {len(ica.exclude)} ICs: {ica.exclude}")
    ## save the recontructed data
    epochs_recon.save(opj(OutPath2, subj.split('/')[-1].split('-epo')[0] + '_recon-epo.fif'), overwrite = True)
    # plot epochs to compare the reconstructed data with the original data
    fig = epochs_recon.plot(n_epochs = 20, n_channels = 62, show_scrollbars = False)
    fig.savefig(opj(OutPathSub, SubNum + '_epochs_recon.png'))
    plt.close('all')
    del epochs_recon, ica, fig
