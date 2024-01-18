from os.path import join as opj
from glob import glob as gg
import os, time, random
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
mpl.use('Agg')
import mne
from mne.time_frequency import AverageTFR, EpochsTFR, read_tfrs
from mne import create_info, read_epochs
from mne.decoding import (Scaler, Vectorizer, CSP,
                         SlidingEstimator, GeneralizingEstimator, cross_val_multiscore)
from sklearn.svm import LinearSVC
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold, 
                                     cross_val_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.model_selection import permutation_test_score
random.seed(int(time.time()))
print(mne.sys_info())
############
def epochs_hilbert(epc, frequencies, times, bsl, envelope=False, csd=False, n_jobs=1):
    # Hilbert transformation on epochs data
    t1, t2 = times
    f1, f2 = frequencies
    epc1 = epc.copy()
    # Laplacian (Current Source Density) spatial filter
    if csd:
        from mne.preprocessing import compute_current_source_density
        epc1 = compute_current_source_density(epc1)
    epc1.filter(f1, f2, fir_design='firwin', n_jobs=n_jobs)
    # Hilbert transformation
    epc1.apply_hilbert(envelope=envelope, n_jobs=n_jobs)
    # baseline correction
    if bsl[0] != bsl[1]: epc1.apply_baseline(bsl)
    # compute power
    if not envelope:
        hilb_data = epc1.crop(tmin=t1, tmax=t2, include_tmax = False).get_data()
        hilb_data1 = hilb_data * hilb_data.conj()
        epc1._data = hilb_data1.astype(float)
    return epc1
############
import pickle
def save_dict(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_dict(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
############
def extract_data(epc, condition_head='speech_act_type'):
    X = []; y = []; runs = []
    for ci in epc.event_id.keys():
        epc1 = epc[ci]
        for rr in epc.metadata.run.unique():
            epc2 = epc1['run == %i'%rr].copy()
            X1 = epc2._data.mean(axis=(0,2))
            X.append(X1)
            y.extend(list(epc2.metadata[condition_head].unique()))
            runs.extend([rr])
    X = np.squeeze(np.vstack(X))
    return X, y, runs
########################
ptimes = 1000
ptimes1 = 100000
RootPath = '~/Experiments/SPF'
pars = np.loadtxt(opj(RootPath,'batch/classifier/featsel_GD_ind.txt'),dtype=str)
lpf = 30
featsel = 100
thr = 100
(b1, b2) = (-0.1, 0.)
n_jobs = 10
cscore = 'roc_auc'
lap_csd = 0
n_splits = 4; n_repeats = 1
OutPath = opj(RootPath, 'output', 'classifer_GD',
              'hilbert_run_%s'%cscore,
              'fs%i_lap%i'%(featsel,int(lap_csd)))
if not os.path.exists(OutPath): os.makedirs(OutPath)
print(OutPath)

SubEx = ['SPF131','SPF132','SPF136']
# time wndow onset and offset
Freqs = {'mu':(8,12), 'beta':(18,25)}
# conditions
Contr = {'PrmAns1':['Ans1','Prm'],'ReqAns2':['Ans2','Req']}
# critical windows
Crts = {'crt3': (0.8*2+0.4*2, 0.8*3+0.4*2),
        'crt3b': (0.8*3+0.4*2, 0.8*3+0.4*3),
        'crt4': (0.8*3+0.4*3, 0.8*4+0.4*3)
        }
# define ROIs
Rois = {
        'fronto-central': ['F7','F5','F3','F1','Fz','F2','F4','F6','F8',
                    'FC5', 'FC1', 'FC2', 'FC6', 'FC3', 'FC4', 'FCz',
                    'C5','C3','C1','Cz','C2','C4','C6'],
        'parieto-occipital':['P7', 'P3', 'Pz', 'P4', 'P8',
                            'P5', 'P1', 'P2', 'P6',
                            'PO9','PO7','PO3','POz','PO4',
                            'PO8','PO10','O1','Oz','O2']
                     }
DataPath = opj(RootPath,'output','step_3_%ilowpass_thr%i'%(lpf,thr))
SubjFiles = gg(opj(DataPath,'SPF*0.1-NoneHzfiltered_caref_recon_rej-epo.fif'))
SubjFiles = [subj for subj in SubjFiles if subj.split('/')[-1].split('_')[0] not in SubEx]
SubjFiles.sort()

outfile = opj(OutPath,'GD_base%s-%s_perm%s-%s'%
              (str(b1), str(b2), str(ptimes), str(ptimes1)))
if not os.path.exists(outfile+'.pkl'):
    Scores = {CrtWin: {FrqNum: {CndNum: {RoiNum: [] for RoiNum in Rois.keys()}
                            for CndNum in Contr.keys()}
                            for FrqNum in Freqs.keys()}
                            for CrtWin in Crts.keys()}
    # time window loop
    for CrtWin, (t1, t2) in Crts.items():
        t1 = np.round(t1, 1); t2 = np.round(t2, 1)
        # frequency level loop
        for FrqNum, (f1,f2) in Freqs.items():
            if os.path.exists(outfile+'.pkl'): continue
            # pairwise classification level loop
            for CndNum, cnds in Contr.items():
                if os.path.exists(outfile+'.pkl'): continue
                X=[]; y=[]; runs=[]
                # subject level loop
                for sid, subj in enumerate(SubjFiles):
                    SubNum = subj.split('/')[-1].split('_')[0]
                    epc = read_epochs(subj, preload=True)
                    epc = epc[cnds]
                    if CrtWin in ['crt34', 'crt3', 'crt3b', 'crt4']:
                        (bs1, bs2) = (Crts[CrtWin][0]+b1, Crts[CrtWin][0]+b2)
                    else:
                        (bs1, bs2) = (b1, b2)
                    epc = epochs_hilbert(epc, (f1,f2), (t1, t2), (bs1, bs2), envelope=False, csd=lap_csd, n_jobs=n_jobs)
                    if not 'channels' in locals().keys(): channels = epc.ch_names
                    X1, y1, runs = extract_data(epc, condition_head='speech_act_type')
                    print(X1.shape, len(y1), len(runs))
                    # Assemble the classifier using scikit-learn pipeline
                    fml = LinearSVC(max_iter=100000)
                    if featsel == 100:
                        clf = make_pipeline(Scaler(scalings='mean'),
                                            Vectorizer(),
                                            fml
                                            )
                    else:
                        clf = make_pipeline(SelectPercentile(f_classif, percentile=featsel),
                                            Scaler(scalings='mean'),
                                            Vectorizer(),
                                            fml
                                            )
                    le = LabelEncoder()
                    # ROI loop
                    for RoiNum, chans in Rois.items():
                        if chans==[]:
                            X2 = X1.copy()
                        else:
                            X2 = X1.copy()[:,[ch in chans for ch in channels]]
                        y2 = le.fit_transform(y1)
                        fold_split = StratifiedKFold(n_splits=n_splits)
                        scr, perm_scores, pvalue = permutation_test_score(
                                                        clf, X2, y2, scoring=cscore, 
                                                        cv=fold_split, groups=runs,
                                                        n_permutations=ptimes-1,
                                                        n_jobs=n_jobs,
                                                        random_state=int(time.time())
                                                        )
                        print(CrtWin, SubNum, FrqNum, CndNum, RoiNum, np.round(scr,2), np.round(pvalue,3))
                        Scores[CrtWin][FrqNum][CndNum][RoiNum].append(np.array([scr]+list(perm_scores))[np.newaxis,])
                for RoiNum, chans in Rois.items():
                    Scores[CrtWin][FrqNum][CndNum][RoiNum] = np.vstack(Scores[CrtWin][FrqNum][CndNum][RoiNum])
    save_dict(Scores, outfile)

#############################
if not 'Scores' in locals(): Scores = load_dict(outfile)
outtxt = opj(OutPath,'GD_base%s-%s_perm%s-%s.txt'%
              (str(b1), str(b2), str(ptimes), str(ptimes1)))
outnull = opj(OutPath,'GD_base%s-%s_perm%s-%s_group'%
              (str(b1), str(b2), str(ptimes), str(ptimes1)))
results = {'CrtWin':[], 'FrqNum':[], 'CndNum':[], 'RoiNum':[], cscore:[], 'p':[]}
nulls = {CrtWin: {FrqNum: {CndNum: {RoiNum: [] for RoiNum in Rois.keys()}
                    for CndNum in Contr.keys()}
                    for FrqNum in Freqs.keys()}
                    for CrtWin in Crts.keys()}
# time window loop
for CrtWin, (t1, t2) in Crts.items():
    # frequency level loop
    for FrqNum, (f1,f2) in Freqs.items():
        # pairwise classification level loop
        for CndNum, cnds in Contr.items():
            scr_null = []
            for RoiNum, chans in Rois.items():
                scrs = Scores[CrtWin][FrqNum][CndNum][RoiNum]
                scr_obs = np.mean(scrs[:,0], axis=0)
                scr_null = [np.mean([scrs[ei, random.randint(1, scrs.shape[1]-1)]
                                        for ei in range(scrs.shape[0])
                                        ]) for pt in range(ptimes1-1)]
                scr_null = [scr_obs] + scr_null
                nulls[CrtWin][FrqNum][CndNum][RoiNum] = scr_null
                pv = np.mean(scr_obs <= scr_null)
                results['CrtWin'].extend([CrtWin])
                results['FrqNum'].extend([FrqNum])
                results['CndNum'].extend([CndNum])
                results['RoiNum'].extend([RoiNum])
                results[cscore].extend([scr_obs])
                results['p'].extend([pv])
save_dict(nulls, outnull)
results = pd.DataFrame(results)
results.to_csv(outtxt, sep='\t', header=True, index=False)
