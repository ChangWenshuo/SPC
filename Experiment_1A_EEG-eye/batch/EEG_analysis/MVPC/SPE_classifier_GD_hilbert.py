from os.path import join as opj
from glob import glob as gg
import os, time, random
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
mpl.use('Agg')
from mne import sys_info
from mne import read_epochs
from mne.decoding import Scaler, Vectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import permutation_test_score
random.seed(int(time.time()))
print(sys_info())
############
def epochs_hilbert(epc, frequencies, bsl, envelope=False, csd=False, n_jobs=1):
    # Hilbert transformation on epochs data
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
        hilb_data = epc1.get_data()
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
def extract_data(epc, condition_head='speech_act_type', time_max='Time34',
                        start_fixation='StartTime34', end_fixation='EndTime34', 
                        start_blink='StartBlink34', end_blink='EndBlink34', eyethr = 60):
    X = []; y = []; runs = []
    for ci in epc.event_id.keys():
        epc1 = epc[ci]
        for rr in epc.metadata.run.unique():
            epc2 = epc1['run == %i and GD%s >= %i'%(rr, time_max.split('Time')[-1], eyethr)].copy()
            X1 = []
            for ei in range(len(epc2)):
                tmx = epc2[ei].metadata[time_max].copy().reset_index(drop=True)[0]
                if not (start_blink == '' or end_blink == ''):
                    bst = epc2[ei].metadata[start_blink].copy().reset_index(drop=True)[0]
                    if bst == '*': 
                        twin = [(0, tmx)]
                    else:
                        bst = bst.split(',')
                        bst = [int(bb) for bb in bst if int(bb) < tmx]
                        fst1 = []; fst2 = []
                        if not bst == []:
                            ben = epc2[ei].metadata[end_blink].copy().reset_index(drop=True)[0].split(',')
                            ben = [int(bb) for bb in ben if int(bb) < tmx]
                            fst = epc2[ei].metadata[start_fixation].copy().reset_index(drop=True)[0].split(',')
                            for bb in ben:
                                fst11 = [int(fs) for fs in fst if bb <= int(fs)]
                                fst1.extend([fst11[0]])
                            fen = epc2[ei].metadata[end_fixation].copy().reset_index(drop=True)[0].split(',')
                            for bb in bst:
                                fst22 = [int(fs) for fs in fen if int(fs) <= bb]
                                fst2.extend([fst22[-1]])
                        twin1 = [0] + fst1
                        twin2 = fst2 + [tmx]
                        twin = [(twin1[t1], twin2[t1]) for t1 in range(len(twin1))]
                elif not (start_fixation == '' or end_fixation == ''):
                    fst = epc2[ei].metadata[start_fixation].copy().reset_index(drop=True)[0].split(',')
                    fen = epc2[ei].metadata[end_fixation].copy().reset_index(drop=True)[0].split(',')
                    twin = [(int(fst[t1]), int(fen[t1])) for t1 in range(len(fst))]
                else:
                    twin = [(0, tmx)]
                if twin == []: continue
                X11 = np.vstack([epc2[ei].copy().crop(tmin=twin[tt][0]/1000, 
                                                        tmax=twin[tt][1]/1000)
                                ._data.mean(axis=(2))
                            for tt in range(len(twin))]).mean(axis=0)[np.newaxis,]
                X1.append(X11)
            X1 = np.vstack(X1).mean(axis=0)
            X.append(X1)
            y.extend(list(epc2.metadata[condition_head].unique()))
            runs.extend([rr])
    X = np.squeeze(np.vstack(X))
    return X, y, runs
########################
ptimes = 1000
ptimes1 = 100000
RootPath = '~/Experiments/SPE'
lpf = 30
featsel = 100
thr = 100
(b1, b2) = (-0.1, 0.)
n_jobs = 32
cscore = 'roc_auc'
lap_csd = 0
n_splits = 4; n_repeats = 1
OutPath = opj(RootPath, 'output', 'classifer_GD',
              'hilbert_run_%s'%cscore,
              'fs%s_lap%i'%(str(featsel),int(lap_csd)))
if not os.path.exists(OutPath): os.makedirs(OutPath)
print(OutPath)

SubEx = ['SPE106','SPE142']
# time wndow onset and offset
Freqs = {'alpha':(8,12), 'beta':(18,25)}
# conditions
Contr = {'PrmAns1':['Ans1','Prm'],'ReqAns2':['Ans2','Req']}
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
SubjFiles = gg(opj(DataPath,'SPE*_0.1-NoneHzfiltered_caref_recon_rej-epo.fif'))
SubjFiles = [subj for subj in SubjFiles if subj.split('/')[-1].split('_')[0] not in SubEx]
SubjFiles.sort()

outfile = opj(OutPath,'GD_base%s-%s_perm%s-%s'%
              (str(b1), str(b2), str(ptimes),str(ptimes1)))
if not os.path.exists(outfile+'.pkl'):
    Scores = {FrqNum: {CndNum: {RoiNum: [] for RoiNum in Rois.keys()}
                        for CndNum in Contr.keys()}
                        for FrqNum in Freqs.keys()}
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
                (bs1, bs2) = (b1,b2)
                epc = epochs_hilbert(epc, (f1,f2), (bs1, bs2), envelope=False, csd=lap_csd, n_jobs=n_jobs)
                if not 'channels' in locals().keys(): channels = epc[cnds[0]].ch_names
                X1, y1, runs = extract_data(epc, condition_head='speech_act_type_x',
                                                start_blink='',
                                                eyethr = 80
                                                )
                print(X1.shape, len(y1), len(runs))
                # Assemble the classifier using scikit-learn pipeline
                fml = LinearSVC(max_iter=100000)
                clf = make_pipeline(Scaler(scalings='mean'),
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
                                                    n_permutations=ptimes,
                                                    n_jobs=n_jobs,
                                                    random_state=int(time.time()))
                    print(SubNum, FrqNum, CndNum, RoiNum, np.round(scr,2), np.round(pvalue,3))
                    Scores[FrqNum][CndNum][RoiNum].append(np.array([scr]+list(perm_scores))[np.newaxis,])
            for RoiNum, chans in Rois.items():
                Scores[FrqNum][CndNum][RoiNum] = np.vstack(Scores[FrqNum][CndNum][RoiNum])
    save_dict(Scores, outfile)
#############################
if not 'Scores' in locals(): Scores = load_dict(outfile)
outtxt = opj(OutPath,'GD_base%s-%s_perm%s-%s.txt'%
              (str(b1), str(b2), str(ptimes), str(ptimes1)))
outnull = opj(OutPath,'GD_base%s-%s_perm%s-%s_group'%
              (str(b1), str(b2), str(ptimes), str(ptimes1)))
results = {'FrqNum':[], 'CndNum':[], 'RoiNum':[], cscore:[], 'p':[]}
nulls = {FrqNum: {CndNum: {RoiNum: [] for RoiNum in Rois.keys()} 
                     for CndNum in Contr.keys()}
                       for FrqNum in Freqs.keys()}
# frequency level loop
for FrqNum, (f1,f2) in Freqs.items():
    # pairwise classification level loop
    for CndNum, cnds in Contr.items():
        scr_null = []
        for RoiNum, chans in Rois.items():
            scrs = Scores[FrqNum][CndNum][RoiNum]
            scr_obs = np.mean(scrs[:,0], axis=0)
            scr_null = [np.mean([scrs[ei, random.randint(1, scrs.shape[1]-1)]
                                     for ei in range(scrs.shape[0])
                                     ]) for pt in range(ptimes1-1)]
            scr_null = [scr_obs] + scr_null
            nulls[FrqNum][CndNum][RoiNum] = scr_null
            pv = np.mean(scr_obs <= scr_null)
            results['FrqNum'].extend([FrqNum])
            results['CndNum'].extend([CndNum])
            results['RoiNum'].extend([RoiNum])
            results[cscore].extend([scr_obs])
            results['p'].extend([pv])
save_dict(nulls, outnull)
results = pd.DataFrame(results)
results.to_csv(outtxt, sep='\t', header=True, index=False)
