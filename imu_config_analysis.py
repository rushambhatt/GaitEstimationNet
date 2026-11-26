#!/usr/bin/env python3
"""Participant-held-out IMU sensor-configuration analysis for the existing 2-head CNN.

Same architecture: Conv1d(32,k7) -> ReLU -> MaxPool1d(k2,s2) ->
Conv1d(64,k5) -> ReLU -> Flatten -> Dense(32) -> ReLU -> Dropout(.25)
-> independent left/right two-logit heads. Window=148, training step=15.

This program performs LOSO evaluation. Its held-out subjects are for CV/analysis,
NOT an untouched final test set. Never optimize hyperparameters using these
held-out results and then call them independent test results.
"""
import argparse
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import Dataset, DataLoader

WINDOW = 148
STEP = 15
AXES = ('acc_x','acc_y','acc_z','gyr_x','gyr_y','gyr_z')
SENSORS = ('LeftShank','RightShank','LeftThigh','RightThigh','Torso','RightFoot','LeftFoot')
SIX = SENSORS[:-1]
SEVEN = SENSORS
SENSOR_LABELS = {'Left foot':'LeftFoot','Right foot':'RightFoot',
                 'Left shank':'LeftShank','Right shank':'RightShank',
                 'Left thigh':'LeftThigh','Right thigh':'RightThigh','Torso':'Torso'}
# Core preset list is derived from the uploaded analysis workbook (not all possible 127 subsets).
WORKBOOK_CONFIGS = [
 'LeftFoot','RightFoot','LeftShank','RightShank','LeftThigh','RightThigh','Torso',
 'LeftFoot+RightFoot','LeftFoot+LeftShank','LeftFoot+LeftThigh','LeftFoot+Torso',
 'RightFoot+RightShank','RightFoot+RightThigh','RightFoot+Torso',
 'LeftShank+RightShank','LeftShank+LeftThigh','LeftShank+Torso',
 'RightShank+RightThigh','RightShank+Torso','LeftThigh+RightThigh',
 'LeftThigh+Torso','RightThigh+Torso','RightFoot+LeftShank','RightFoot+LeftThigh',
 '6IMU','7IMU']


def set_seed(seed, deterministic=False):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
        # Enables deterministic kernels where available, while allowing programs
        # to run on hardware where some operations lack a deterministic variant.
        torch.use_deterministic_algorithms(True, warn_only=True)


class TwoHeadGaitCNN(nn.Module):
    """Identical layer names and dimensions to train_imu_configurable.py."""
    def __init__(self, n_channels):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Conv1d(n_channels,32,kernel_size=7,stride=1), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2,stride=2),
            nn.Conv1d(32,64,kernel_size=5,stride=1),nn.ReLU(),nn.Flatten())
        with torch.no_grad():
            flat_dim=self.shared(torch.zeros(1,n_channels,WINDOW)).shape[1]
        self.dense=nn.Sequential(nn.Linear(flat_dim,32),nn.ReLU(),nn.Dropout(.25))
        self.left_head=nn.Linear(32,2)
        self.right_head=nn.Linear(32,2)
    def forward(self,x):
        z=self.dense(self.shared(x))
        return self.left_head(z),self.right_head(z)


def config_sensors(name):
    if name.lower() in ('6imu','6'): return SIX
    if name.lower() in ('7imu','7'): return SEVEN
    parts=name.split('+')
    if any(p not in SENSORS for p in parts) or len(parts)!=len(set(parts)):
        raise ValueError(f'Bad configuration {name!r}. Valid sensor names: {SENSORS}')
    return tuple(parts)


def participant(path):
    parts=Path(path).stem.split('_')
    if len(parts)<3 or parts[-1].lower()!='labeled' or not re.fullmatch(r'[A-Za-z][A-Za-z0-9-]*',parts[-2]):
        raise ValueError(f'Filename must follow ..._Name_labeled.csv: {path}')
    return parts[-2]


def speed_from_name(path):
    # Date_0_5mph_Name_labeled.csv, Date_1_0_Name_labeled.csv, etc.
    stem=Path(path).stem
    match=re.search(r'(?:^|_)(\d+)_(\d+)(?:mph)?_(?:[A-Za-z][A-Za-z0-9-]*)_labeled$',stem,re.I)
    return float(match.group(1)+'.'+match.group(2)) if match else float('nan')


def file_inventory(paths):
    rows=[]
    for p in paths:
        header=pd.read_csv(p,nrows=0).columns.tolist()
        names=set(header)
        present={s: all(f'{s}_{ax}' in names for ax in AXES) for s in SENSORS}
        rows.append({'file':str(p),'subject':participant(p),'speed_mph':speed_from_name(p),
                     'available_sensors':'+'.join(s for s in SENSORS if present[s]),
                     'missing_sensors':'+'.join(s for s in SENSORS if not present[s]),
                     'has_labels':all(k in names for k in ('L_stance','R_stance')),
                     **{f'has_{s}':present[s] for s in SENSORS}})
    return pd.DataFrame(rows)


def load_record(path,features):
    df=pd.read_csv(path,low_memory=False)
    a=df[features].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=np.float32)
    labels=df[['L_stance','R_stance']].apply(pd.to_numeric,errors='coerce').to_numpy()
    valid_label=np.isin(labels,[0,1]).all(axis=1)
    valid_signal=np.isfinite(a).all(axis=1)
    bad=(~valid_signal).astype(np.int32)
    bad_cum=np.concatenate(([0],np.cumsum(bad)))
    starts=np.arange(0,max(0,len(a)-WINDOW+1),STEP,dtype=np.int64)
    good=(bad_cum[starts+WINDOW]-bad_cum[starts]==0)&valid_label[starts+WINDOW-1]
    starts=starts[good]
    return {'signals':a, 'labels':labels, 'starts':starts,'name':Path(path).name,
            'subject':participant(path),'speed':speed_from_name(path)}


class Windows(Dataset):
    def __init__(self, records,mean,std):
        self.records=records
        self.index=[(i,int(k)) for i,r in enumerate(records) for k in r['starts']]
        self.mean=mean.astype(np.float32)
        self.std=std.astype(np.float32)
    def __len__(self):return len(self.index)
    def __getitem__(self,i):
        j,start=self.index[i]
        rec=self.records[j]
        x=rec['signals'][start:start+WINDOW]
        x=((x-self.mean)/self.std).T.copy()
        y=rec['labels'][start+WINDOW-1].astype(np.int64)
        return torch.from_numpy(x),int(y[0]),int(y[1]),j,start+WINDOW-1


def train_stats(records):
    """Match original training-window normalization without materializing X[N,C,148].

    Each timestamp contributes once for EVERY valid training window containing it,
    exactly as normalize_train_only() in the original trainer (up to FP summation).
    """
    total=squares=None
    n=0
    for r in records:
        starts=r['starts']
        if not len(starts):continue
        sig=r['signals'].astype(np.float64,copy=False)
        diff=np.zeros(len(sig)+1,dtype=np.int64)
        np.add.at(diff,starts,1)
        np.add.at(diff,starts+WINDOW,-1)
        counts=np.cumsum(diff[:-1])
        active=counts>0
        vals=sig[active]
        weights=counts[active].astype(np.float64)
        if total is None:
            total=np.zeros(sig.shape[1]);squares=np.zeros(sig.shape[1])
        total+=(vals*weights[:,None]).sum(axis=0)
        squares+=((vals*vals)*weights[:,None]).sum(axis=0)
        n+=int(weights.sum())
    if not n:raise ValueError('No valid training windows for normalization.')
    mean=total/n;std=np.sqrt(np.maximum(squares/n-mean**2,0))
    return mean.astype(np.float32),np.maximum(std,1e-6).astype(np.float32)


def prediction_metrics(y,p):
    return {'accuracy':float(accuracy_score(y,p)),
            'balanced_accuracy':float(balanced_accuracy_score(y,p)),
            'macro_f1':float(f1_score(y,p,average='macro',zero_division=0)),
            'swing_recall':float(((y==0)&(p==0)).sum()/max(1,(y==0).sum())),
            'stance_recall':float(((y==1)&(p==1)).sum()/max(1,(y==1).sum()))}


def fit_and_score(train_records,test_records,channels,args,seed,device):
    set_seed(seed,args.deterministic)
    mu,std=train_stats(train_records)
    train_ds=Windows(train_records,mu,std);test_ds=Windows(test_records,mu,std)
    if not len(train_ds) or not len(test_ds):raise ValueError('No usable train/test windows')
    train_loader=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,
                            num_workers=args.num_workers,pin_memory=(device.type=='cuda'))
    test_loader=DataLoader(test_ds,batch_size=args.batch_size,shuffle=False,
                           num_workers=args.num_workers,pin_memory=(device.type=='cuda'))
    model=TwoHeadGaitCNN(channels).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=args.learning_rate)
    lossfn=nn.CrossEntropyLoss()
    losses=[]
    for ep in range(args.epochs):
        model.train();total=0.0;count=0
        for xb,yl,yr,_,_ in train_loader:
            xb=xb.to(device,non_blocking=True);yl=yl.to(device);yr=yr.to(device)
            opt.zero_grad(set_to_none=True)
            left,right=model(xb)
            loss=lossfn(left,yl)+lossfn(right,yr)
            loss.backward();opt.step()
            total+=float(loss.item())*len(xb);count+=len(xb)
        losses.append(total/max(1,count))
        if ep==0 or (ep+1)%max(1,args.epochs//5)==0:
            print(f'    epoch {ep+1:3d}/{args.epochs} loss={losses[-1]:.4f}',flush=True)
    model.eval();pred=[]
    with torch.inference_mode():
        for xb,yl,yr,ridx,ends in test_loader:
            left,right=model(xb.to(device,non_blocking=True))
            lpred=left.argmax(dim=1).cpu().numpy();rpred=right.argmax(dim=1).cpu().numpy()
            for k in range(len(yl)):
                rec=test_records[int(ridx[k])]
                pred.append({'subject':rec['subject'],'file':rec['name'],
                    'speed_mph':rec['speed'],'window_end_row':int(ends[k]),
                    'L_true':int(yl[k]),'R_true':int(yr[k]),
                    'L_pred':int(lpred[k]),'R_pred':int(rpred[k])})
    return pd.DataFrame(pred),losses


def results_from_predictions(df,config,held_out,run_seed):
    result={'config':config,'held_out_subject':held_out,'seed':run_seed,
            'n_test_windows':len(df)}
    for leg in ('L','R'):
        for name,val in prediction_metrics(df[f'{leg}_true'].to_numpy(),df[f'{leg}_pred'].to_numpy()).items():
            result[f'{leg}_{name}']=val
    return result


def run(args):
    import glob
    paths=sorted({Path(p).resolve() for pattern in args.data for p in glob.glob(pattern,recursive=True)})
    if not paths:raise ValueError('No CSV files matched --data pattern.')
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    inventory=file_inventory(paths)
    inventory.to_csv(output/'recording_inventory.csv',index=False)
    print('Detected subjects:',sorted(inventory.subject.unique()))
    print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not available (CPU fallback)')
    device=torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    if args.require_cuda and device.type!='cuda':raise RuntimeError('CUDA required but unavailable. Install NVIDIA-supported PyTorch.')
    configs=WORKBOOK_CONFIGS if args.workbook_configs else args.configs
    if not configs:raise ValueError('Select --configs or --workbook-configs')
    for name in configs:config_sensors(name)
    if args.test_subject:
        assert args.test_subject in set(inventory.subject),'--test-subject not found in filenames'
        print('Untouched final test subject excluded from ALL runs:',args.test_subject)
    matched_file_set=None
    if args.matched_trials:
        matched_file_set=set(inventory.loc[inventory.has_labels,'file'])
        for cfg in configs:
            selected=config_sensors(cfg)
            eligible_files=inventory[inventory.has_labels]
            for sensor in selected:
                eligible_files=eligible_files[eligible_files[f'has_{sensor}']]
            matched_file_set &= set(eligible_files.file)
        if args.test_subject:
            matched_file_set={f for f in matched_file_set
                              if inventory.loc[inventory.file.eq(f),'subject'].iloc[0] != args.test_subject}
        print(f'Matched-trials mode: {len(matched_file_set)} recordings eligible for every requested configuration')
    all_rows=[];speed_rows=[];prediction_dir=output/'predictions';prediction_dir.mkdir(exist_ok=True)
    eligibility=[]
    for config in configs:
        names=config_sensors(config)
        eligible=inventory[inventory.has_labels].copy()
        for s in names:eligible=eligible[eligible[f'has_{s}']]
        eligible=eligible[eligible.subject.ne(args.test_subject)] if args.test_subject else eligible
        if matched_file_set is not None:
            eligible=eligible[eligible.file.isin(matched_file_set)]
        subject_names=sorted(eligible.subject.unique())
        excluded=inventory[~inventory.file.isin(eligible.file)]
        for _,r in excluded.iterrows():
            eligibility.append({'config':config,'file':r['file'],'subject':r['subject'],
                                'status':'excluded','reason':'missing selected sensor(s)/labels or reserved final-test subject'})
        for _,r in eligible.iterrows():
            eligibility.append({'config':config,'file':r['file'],'subject':r['subject'],'status':'included','reason':''})
        if len(subject_names)<2:
            print(f'CONFIG {config}: SKIPPED ({len(subject_names)} eligible participant(s))')
            continue
        print(f'\nCONFIG {config}: {len(names)} sensor(s), {len(eligible)} files, {len(subject_names)} subjects')
        # Each configuration trains from scratch, on the same fixed architecture.
        features=[f'{sensor}_{axis}' for sensor in names for axis in AXES]
        records=[load_record(Path(p),features) for p in eligible.file]
        if args.comparison_subjects:
            wanted=set(args.comparison_subjects)
            records=[r for r in records if r['subject'] in wanted]
        held_subjects=sorted(set(r['subject'] for r in records))
        if len(held_subjects)<2:continue
        for held in held_subjects:
            tr=[r for r in records if r['subject']!=held]
            te=[r for r in records if r['subject']==held]
            for seed in args.seeds:
                print(f'  held out={held} seed={seed} train_windows={sum(len(r["starts"]) for r in tr)}')
                try:
                    pred,losses=fit_and_score(tr,te,len(features),args,seed,device)
                except ValueError as exc:
                    print('  SKIP:',exc);continue
                pred.insert(0,'config',config);pred.insert(2,'seed',seed)
                all_rows.append(results_from_predictions(pred,config,held,seed))
                for speed,g in pred.groupby('speed_mph',dropna=False):
                    one=results_from_predictions(g,config,held,seed)
                    one['speed_mph']=speed;speed_rows.append(one)
                if args.save_predictions:
                    safe=re.sub('[^A-Za-z0-9_-]','_',config)
                    pred.to_csv(prediction_dir/f'{safe}__{held}__seed{seed}.csv',index=False)
                (prediction_dir/f'{re.sub("[^A-Za-z0-9_-]","_",config)}__{held}__seed{seed}_loss.json').write_text(json.dumps(losses))
                if device.type=='cuda':torch.cuda.empty_cache()
    pd.DataFrame(eligibility).to_csv(output/'recording_eligibility.csv',index=False)
    if not all_rows:raise RuntimeError('No experiments completed. Check inventory and labels.')
    folds=pd.DataFrame(all_rows)
    folds.to_csv(output/'held_out_participant_metrics.csv',index=False)
    pd.DataFrame(speed_rows).to_csv(output/'metrics_by_speed.csv',index=False)
    # Critical: mean and SD are computed across independent SUBJECTS, not across windows.
    if len(args.seeds)>1:
        per_subject=folds.groupby(['config','held_out_subject'],as_index=False).mean(numeric_only=True)
    else:
        per_subject=folds.copy()
    metric_cols=[x for x in per_subject.columns if x.startswith(('L_','R_'))]
    summary=per_subject.groupby('config',as_index=False).agg(
        n_subjects=('held_out_subject','nunique'),
        **{f'{k}_mean':(k,'mean') for k in metric_cols},
        **{f'{k}_sd':(k,'std') for k in metric_cols})
    summary.to_csv(output/'configuration_summary.csv',index=False)
    wide=[]
    for config,g in per_subject.groupby('config'):
        row={'Configuration':config,'# IMUs':len(config_sensors(config))}
        for _,v in g.iterrows():
            row[f'{v.held_out_subject} Left leg']=v['L_accuracy']
            row[f'{v.held_out_subject} Right leg']=v['R_accuracy']
        row['Left leg Mean']=g['L_accuracy'].mean()
        row['Left leg SD (pp)']=100*g['L_accuracy'].std(ddof=1)
        row['Right leg Mean']=g['R_accuracy'].mean()
        row['Right leg SD (pp)']=100*g['R_accuracy'].std(ddof=1)
        wide.append(row)
    pd.DataFrame(wide).to_csv(output/'accuracy_table_both_legs.csv',index=False)
    (output/'analysis_settings.json').write_text(json.dumps({
        'configs':configs,'epochs':args.epochs,'seeds':args.seeds,'batch_size':args.batch_size,
        'learning_rate':args.learning_rate,'window':WINDOW,'step':STEP,
        'cuda_used':device.type=='cuda','test_subject_excluded':args.test_subject,
        'comparison_subjects':args.comparison_subjects,'matched_trials':args.matched_trials,
        'note':'LOSO CV estimates, not untouched final-test results; comparison requires matching eligible subjects/trials.'
    },indent=2))
    print('\nWrote results to',output.resolve())
    print(summary[['config','n_subjects','L_accuracy_mean','R_accuracy_mean']].to_string(index=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',nargs='+',required=True,help='Quoted glob(s), e.g. "data/*_labeled.csv"')
    p.add_argument('--configs',nargs='+',default=['LeftShank','LeftShank+RightShank','6IMU','7IMU'])
    p.add_argument('--workbook-configs',action='store_true',help='Run the sensor configurations listed in teammate workbook')
    p.add_argument('--matched-trials',action='store_true',help='Restrict all configurations to identical recordings with EVERY selected sensor available')
    p.add_argument('--comparison-subjects',nargs='+',default=None,help='Restrict analysis to common subjects')
    p.add_argument('--test-subject',default=None,help='Exclude this person from ALL model selection; does NOT score final test')
    p.add_argument('--epochs',type=int,default=50)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--learning-rate',type=float,default=0.001)
    p.add_argument('--seeds',nargs='+',type=int,default=[2001])
    p.add_argument('--num-workers',type=int,default=0)
    p.add_argument('--output',default='analysis_results')
    p.add_argument('--save-predictions',action='store_true')
    p.add_argument('--cpu',action='store_true')
    p.add_argument('--require-cuda',action='store_true')
    p.add_argument('--deterministic',action='store_true')
    run(p.parse_args())

if __name__=='__main__':main()
