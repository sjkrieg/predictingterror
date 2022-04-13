"""
Author: Steven Krieg (skrieg@nd.edu)
Dependencies: Python 3.7, sklearn (0.22), pandas (0.24), imblearn (0.6.2), tqdm (4.45.0)
Description:
    Trains a Random Forest model on news features from GDELT to predict
    the occurrence of terrorist attacks.
    
    Input is gdelt features as a .csv file and gtd labels as a .csv file.
    
    Output is 2 .csv files: one for the 5-fold cross validation result, and one for daily predictions.
"""

import argparse
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import utils # custom utils
from datetime import datetime
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from time import perf_counter
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-if1', '--gdelt', default='/afs/crc.nd.edu/user/s/skrieg/Private/psicorp/dat/2019-11-01/gdelt-features.csv')
parser.add_argument('-if2', '--gtd', default='/afs/crc.nd.edu/user/s/skrieg/Private/psicorp/gtd/gtd_relevant_events.csv')
# max delta to consider for observation windows
parser.add_argument('-d', '--maxdelta', type=int, default=14)
# number of workers to utilize in random forest training
parser.add_argument('-c', '--cpus', type=int, default=mp.cpu_count())
# locations to test
parser.add_argument('-l', '--locs', nargs='+', type=lambda x: x.upper(), default=['NY','CA','TX','FL','WA'])
# whether to train/test all locs as one group
parser.add_argument('-g', '--grouptest', action='store_true')
# supplemental training locs
parser.add_argument('-t', '--trainlocs', nargs='+', type=lambda x: x.upper(), default=None)
# if a state has too few events, we aggregate events from other states
parser.add_argument('-e', '--eventthreshold', type=int, default=10)
# bitmap for feature inclusion in pattern cameo_counts, cameo_toneavg, theme_counts, theme_toneavg
# e.g. to run without cameo_counts, pass '0111'
parser.add_argument('-x', '--includefeatures', default='1111')
# prediction window
parser.add_argument('-w', '--window', type=int, default=1)
# number of test folds
parser.add_argument('-f', '--folds', type=int, default=5)
# number of estimators for rf model
parser.add_argument('-n', '--nestimators', type=int, default=3000)
# random seeds
parser.add_argument('-r', '--rseeds', nargs='+', type=int, default=[777, 11767, 123818, 1, 200])
# whether to balance classes via class weights. for RF we prefer using SMOTE.
parser.add_argument('-b', '--balance', action='store_true')
# debug mode loads a smaller version of gdelt and prints more messages
parser.add_argument('-q', '--debug', action='store_true')
# whether to drop melee/unarmed events from the gtd
parser.add_argument('-m', '--dropmelee', action='store_true', default=False)
# whether to drop the training location from testing, i.e. train on california but test on another state
parser.add_argument('-z', '--droptestloc', action='store_true', default=False)
args = parser.parse_args()
deltas = range(1, args.maxdelta+1)

# input/output locations
if args.debug:
    inf_gdelt = '/afs/crc.nd.edu/user/s/skrieg/Private/psicorp/dat/2019-11-01/gdelt-sample.csv'
else:
    inf_gdelt = args.gdelt
inf_gtd = args.gtd
out_dir = 'results/' + datetime.now().strftime('%Y%m%d') + '/'
otf_name = 'rf_d{:02d}.csv'.format(args.maxdelta)

df_probs = []
cols_to_drop = ['Date','Loc']
results = {'Loc':[], 'Fold': [], 'Seed': [], 'AUC': [], 'APR': []}

print('Executing the following configuration.')
for k, v in vars(args).items():
    print('\t-->{:>20} = {}'.format(k,v))

print()
print('-'*20)
print('Reading GDELT input from {}...'.format(inf_gdelt), end='')
gdelt = pd.read_csv(inf_gdelt)
if args.debug: gdelt = gdelt[gdelt.columns[:80]]
if args.includefeatures[0] == '0':
    print('Dropping cameo counts from feature space...')
    gdelt = gdelt.drop([c for c in gdelt.columns if c.startswith('cameo_') and c.endswith('_count')] , axis=1)
if args.includefeatures[1] == '0':
    print('Dropping cameo toneavg from feature space...')
    gdelt = gdelt.drop([c for c in gdelt.columns if c.startswith('cameo_') and c.endswith('_toneavg')] , axis=1)
if args.includefeatures[2] == '0':
    print('Dropping theme counts from feature space...')
    gdelt = gdelt.drop([c for c in gdelt.columns if c.startswith('theme_') and c.endswith('_count')] , axis=1)
if args.includefeatures[3] == '0':
    print('Dropping theme toneavg from feature space...')
    gdelt = gdelt.drop([c for c in gdelt.columns if c.startswith('theme_') and c.endswith('_toneavg')] , axis=1)

# create list of remaining features
features = [c for c in gdelt.columns if c not in cols_to_drop and c != 'label']
print('Done! Processed {} records and {} features from gdelt'.format(len(gdelt), len(features)))
print('Reading GTD input from {}...'.format(inf_gtd), end='')
gtd = pd.read_csv(inf_gtd, encoding='ISO-8859-1', dtype=object, keep_default_na=False)
print('Done!\n')
print('-'*20)
print('Labelling features...')
utils.label_features(gdelt, gtd, drop=args.dropmelee)
print('Done!\n')
print('-'*20)

# don't test or train on states with zero events
print('Dropping states with zero events...', end='')
event_counts = gdelt[['Loc','label']].groupby(['Loc']).agg('sum')['label']
drop_states = event_counts[lambda x: x == 0].index
gdelt = gdelt.loc[~gdelt['Loc'].isin(drop_states)]
print('Dropped states {}.'.format(drop_states.to_numpy()))

# get final list of locations
locs = args.locs if args.locs else sorted(gdelt['Loc'].unique())

print('Aggregating dates with prediction window {}'.format(args.window))
if args.window > 1: 
    gdelt = utils.agg_dates(gdelt, features, window=args.window)
print('Testing on locs:')
for loc in locs: 
    print('\t--> {} ({} events)'.format(loc, event_counts.at[loc]))

# main loop
for loc in locs:
    print('\n{}\nGenerating model for {}...'.format('-' * 20, loc if not args.grouptest else '/'.join(locs)))
    
    # several possible scenarios:
    # 1) we manually specified training locations
    # 2) we are performing a test on a collective group of states
    # 3) we are performing a normal experiment (single state train/test)
    if args.trainlocs:
        train_locs = args.trainlocs 
    elif args.grouptest:
        train_locs = locs
    else:
        train_locs = []
    if train_locs:
        print('Training model on {}'.format(train_locs))
        
    dates_to_drop = [] # not used
    # df_loc contains observations for the current location; df_nloc is for neighboring or supplemental locations
    df_loc = gdelt.loc[(gdelt['Loc'] == loc) & (~gdelt['Date'].isin(dates_to_drop))].reset_index(drop=True)
    df_nloc = gdelt.loc[(gdelt['Loc'].isin(train_locs)) & (gdelt['Loc'] != loc)]
    print('\t--> {} examples in location; {} in neighbor locations'.format(len(df_loc), len(df_nloc)))
    
    # train/test split and adjustment
    folds = KFold(args.folds, shuffle=False).split(df_loc)
    folds = utils.adjust_folds(folds, np.max(deltas))
    
    for i, fold in enumerate(folds):
        print('\nPreparing {} fold {} / {}...'.format(loc if not args.grouptest else '/'.join(locs), i + 1, args.folds))
        train_idx, test_idx = fold

        # if the training set doesn't contain enough events, we aggregate from neighbor locations
        if (df_loc.iloc[train_idx]['label'].sum() < args.eventthreshold) and not args.trainlocs and not args.grouptest:
            train_dates = df_loc.iloc[train_idx]['Date'].unique()
            train_locs = utils.agg_neighbors(gdelt.loc[gdelt['Date'].isin(train_dates)], loc, args.eventthreshold, normalize=True)
            df_nloc = gdelt.loc[(gdelt['Loc'].isin(train_locs)) & (gdelt['Loc'] != loc)]
        
        # if we have multiple locations for the training set, we combine them here
        if args.droptestloc and not df_nloc.empty:
            df_train = df_nloc.groupby('Loc', as_index=False).apply(lambda x: x.iloc[train_idx]).reset_index(drop=True)
        else:
            df_train = pd.concat([df_nloc.groupby('Loc', as_index=False).apply(lambda x: x.iloc[train_idx]), 
                                  df_loc.iloc[train_idx]]).sort_values(by=['Loc','Date'])
        
        # if we have multiple locations for the testing set, combine them here
        if args.grouptest:
            df_test = pd.concat([df_loc.loc[test_idx], df_nloc.groupby('Loc', as_index=False).apply(lambda x: x.iloc[test_idx]).reset_index(drop=True)])
        else: # otherwise test on a single location
            df_test = df_loc.loc[test_idx]
        print('\t--> {} training examples (dates {}-{}); {} testing (dates {}-{})'.format(len(df_train), df_train['Date'].min(), df_train['Date'].max(),len(df_test), df_test['Date'].min(), df_test['Date'].max()))
        
        print('\n\tDetermining optimal deltas from training data using {} workers...'.format(args.cpus))
        # when computing deltas for training set we provide delta_range, not delta_map (see utils.compute_deltas)
        df_train, delta_map = utils.compute_deltas(df_train, args.cpus, delta_range=deltas)
        features = [c for c in df_train.columns if c not in cols_to_drop + ['label']]
        x_train = df_train[features].to_numpy()
        y_train = df_train['label'].to_numpy()
        
        print('\tCalculating deltas for testing data using {} workers...'.format(args.cpus))
        # when computing deltas for testing set we provide delta_map as computed from training set
        df_test, _ = utils.compute_deltas(df_test, args.cpus, delta_map=delta_map)
        x_test = df_test[features].to_numpy()
        y_test = df_test['label']
        if y_test.sum() < 1:
            print('\t--> No events in test set. Skipping fold.')
            continue
        if y_train.sum() < 6:
            print('\t--> Not enough events in training set. Skipping fold.')
            continue
        print('\tData summary:')
        print('\t--> TP representation in training: {} / {} ({:.4f})'.format(y_train.sum(), len(y_train), y_train.sum() / len(y_train)))
        print('\t--> TP representation in testing: {} / {} ({:.4f})'.format(y_test.sum(), len(y_test), y_test.sum() / len(y_test)))
        print('\t--> Delta distribution:', end='\n\t\t')
        delta_dist = pd.DataFrame.from_dict(data=delta_map, orient='index', columns=['Delta', 'Pval']).reset_index()
        print('\n\t\t'.join(delta_dist['Delta'].value_counts(bins=10).sort_index().to_string().split('\n')))
        
        # train the model once for each seed
        for seed in args.rseeds:
            print('\n\tStarting training on seed {}...'.format(seed), end='')
            start = perf_counter()
                
            model = RandomForestClassifier(n_estimators=args.nestimators, 
                                                n_jobs=args.cpus, 
                                                criterion='gini', 
                                                max_depth=int(len(features)**(1/3)),
                                                bootstrap=True,
                                                random_state=seed,
                                                class_weight='balanced' if args.balance else None)
            
            # if args.balance then we use class weights to balance classes
            if args.balance:
                x_train_res, y_train_res = x_train, y_train
            # otherwise use SMOTE (best performance)
            else:
                print('Generating synthetic samples...',end='')
                x_train_res, y_train_res = SMOTE(random_state=seed).fit_resample(x_train, y_train)
            model.fit(x_train_res, y_train_res)
            print('Done! ({:.2f}s)'.format(perf_counter() - start))
            
            #feature importances - not used
            #fimp = pd.Series(data=model.feature_importances_, index=features)
            #print(fimp.sort_values(ascending=False).head(20))
            #print(pd.DataFrame.from_dict(delta_map, columns=['delta','pval'], orient='index').sort_values(by='pval', ascending=True))
            
            # summarize outputs
            probs = pd.Series(model.predict_proba(x_test)[:,-1])
            test_dates = df_test.loc[:,['Loc','Date']]
            test_dates.loc[:,'Pred'] = probs
            test_dates.loc[:,'Seed'] = seed
            test_dates.loc[:,'ytrue'] = y_test
            df_probs.append(test_dates)
            results['Loc'].append(loc if not args.grouptest else '/'.join(locs))
            results['Fold'].append(i + 1)
            results['Seed'].append(seed)
            results['AUC'].append(roc_auc_score(y_test, probs) if df_test['label'].sum() > 0 else np.nan)
            results['APR'].append(average_precision_score(y_test, probs) if df_test['label'].sum() > 0 else np.nan)
            print('\t--> AUC: {:.4f}'.format(results['AUC'][-1]))
            print('\t--> APR: {:.4f}'.format(results['APR'][-1]))
        print('\n*** Average AUC for {} fold {} ({} seeds): {:.4f}'.format(loc if not args.grouptest else '/'.join(locs), i + 1, len(args.rseeds), np.mean(results['AUC'][-len(args.rseeds):])))
        print('*** Average APR for {} fold {} ({} seeds): {:.4f}'.format(loc if not args.grouptest else '/'.join(locs), i + 1, len(args.rseeds), np.mean(results['APR'][-len(args.rseeds):])))
    print('\n*** Average AUC for {} ({} folds, {} seeds): {:.4f} +- {:.4f}'.format(loc if not args.grouptest else '/'.join(locs), args.folds, len(args.rseeds), np.mean(results['AUC'][-(args.folds*len(args.rseeds)):]), np.std(results['AUC'][-(args.folds*len(args.rseeds)):])))
    print('*** Average APR for {} ({} folds, {} seeds): {:.4f} +- {:.4f}'.format(loc if not args.grouptest else '/'.join(locs), args.folds, len(args.rseeds), np.mean(results['APR'][-(args.folds*len(args.rseeds)):]), np.std(results['APR'][-(args.folds*len(args.rseeds)):])))
    
    if args.grouptest: # if testing as group we don't need to repeat
        break
        
print('\nAll locations complete. Saving results to {}...'.format(out_dir + otf_name))
if not os.path.exists(out_dir):
    if not os.path.exists('results/'):
        os.mkdir('results/')
    os.mkdir(out_dir)
summary = pd.DataFrame.from_dict(results)
summary.to_csv(out_dir + otf_name, index=False)
print(summary.groupby('Loc')[['AUC','APR']].agg(['mean','std']))
pd.concat(df_probs).sort_values(by=['Loc','Date','Seed']).to_csv(out_dir + otf_name[:-4] + '_preds.csv', index=False)
print('\nExecution complete!')