"""
Author: Steven Krieg (skrieg@nd.edu)
Dependencies: Python 3.7, tensorflow (2.0), sklearn (0.22), pandas (0.24), imblearn (0.6.2), tqdm (4.45.0)
Description:
    Trains a FFNN model on news features from GDELT to predict
    the occurrence of terrorist attacks.
    
    Input is gdelt features as a .csv file and gtd labels as a .csv file.
    
    Output is 2 .csv files: one for the 5-fold cross validation result, and one for daily predictions.
"""

import argparse
import os
import pandas as pd
import utils # custom utils
import numpy as np
import tensorflow as tf
from datetime import datetime
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import KFold
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, concatenate, Dropout
from tensorflow.keras import regularizers
from tensorflow_addons.losses import SigmoidFocalCrossEntropy
from time import perf_counter
from tqdm import tqdm

parser = argparse.ArgumentParser()
# number of observations to consider for feature feature; input layer will be size delta * nfeatures
parser.add_argument('delta', type=int)
parser.add_argument('-if1', '--gdelt', default='gdelt-features.csv')
parser.add_argument('-if2', '--gtd', default='gtd_relevant_events.csv')
# locations for testing
parser.add_argument('-l', '--locs', nargs='+', type=lambda x: x.upper(), default=['NY','CA','TX','FL','WA'])
# whether to test locations as a group
parser.add_argument('-g', '--grouptest', action='store_true')
# supplemental training locations
parser.add_argument('-t', '--trainlocs', nargs='+', type=lambda x: x.upper(), default=None)
# if a state has too few events, we aggregate events from other states
parser.add_argument('-e', '--eventthreshold', type=int, default=10)
# prediction window
parser.add_argument('-w', '--window', type=int, default=1)
# bitmap for feature inclusion in pattern cameo_counts, cameo_toneavg, theme_counts, theme_toneavg
# e.g. to run without cameo_counts, pass '0111'
parser.add_argument('-x', '--includefeatures', default='1111')
# number of training/test folds
parser.add_argument('-f', '--folds', type=int, default=5)
# set true to use SMOTE, defaults to using a balanced cross entropy
parser.add_argument('-b', '--resampling', default=0, type=int, choices=[-1,0,1,2,3], help='-1 for no none, 0 for none (balanced weights), 1 for SMOTE, 2 for random oversampling, 3 for random under+over')
# number of hidden units
parser.add_argument('-u', '--units', type=int, default=8000)
# number of epochs for training
parser.add_argument('-i', '--epochs', type=int, default=100)
# random seeds for repeat experiments
parser.add_argument('-lr', '--learningrate', type=float, default=.0001)
parser.add_argument('-r', '--rseeds', nargs='+', type=int, default=[777, 11767, 123818, 1, 200])
# debug mode loads a smaller version of gdelt and prints more messages
parser.add_argument('-q', '--debug', action='store_true')
# whether to drop melee/unarmed events from the gtd
parser.add_argument('-m', '--dropmelee', action='store_true', default=False)
# whether to drop the training location from testing, i.e. train on california but test on another state
parser.add_argument('-z', '--droptestloc', action='store_true', default=False)
args = parser.parse_args()
delta = args.delta
n_epochs = args.epochs
num_folds = args.folds

if args.debug:
    inf_gdelt = '/afs/crc.nd.edu/user/s/skrieg/Private/psicorp/dat/2019-11-01/gdelt-sample.csv'
else:
    inf_gdelt = args.gdelt
inf_gtd = args.gtd
out_dir = 'results/' + datetime.now().strftime('%Y%m%d') + '/'
otf_name = 'ff1h_d{:02d}{}_{}.csv'.format(args.delta, '_' + '-'.join(args.locs) if args.grouptest else '', args.resampling)
cols_to_drop = ['Date','Loc']
df_probs = []
results = {'Loc':[], 'Fold': [], 'Seed': [], 'AUC': [], 'APR':[]}

# balanced cross entropy loss function for training
# higher values for alpha mean higher contribution of class 1 to the loss
def balanced_cross_entropy(alpha=.5):
    def balanced_cros_entropy_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        return -K.sum(alpha * K.log(pt_1)) - K.sum((1 - alpha) * K.log(1. - pt_0))
    return balanced_cros_entropy_fixed


print('Executing the following configuration.') 
for k, v in vars(args).items():
    print('\t-->{:>20} = {}'.format(k,v))

# initialize data
print()
print('-'*20)
print('Reading GDELT input from {}...'.format(inf_gdelt), end='')
gdelt = pd.read_csv(inf_gdelt)
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
print('Dropping states with zero events...', end='')
event_counts = gdelt[['Loc','label']].groupby(['Loc']).agg('sum')['label']
drop_states = event_counts[lambda x: x == 0].index
gdelt = gdelt.loc[~gdelt['Loc'].isin(drop_states)]
print('Dropped states {}.'.format(drop_states.to_numpy()))
print('Aggregating dates with prediction window {}'.format(args.window))
if args.window > 1: 
    gdelt = utils.agg_dates(gdelt, features, window=args.window)
    
locs = args.locs if args.locs else sorted(gdelt['Loc'].unique())
print('Testing on locs:')
for loc in locs: 
    print('\t--> {} ({} events)'.format(loc, event_counts.at[loc]))

# normalize df
print('Normalizing input data...', end='')
gdelt[features] = MinMaxScaler().fit_transform(gdelt[features])
print('Done!')

# initialize model architecture
print('Initializing model architecture...', end='')
inputs = []
for j in range(len(features)):
    input = Input(shape=(delta,), name='input_{:03d}'.format(j))
    inputs.append(input)
c = concatenate(inputs)
#d = Dropout(0.3, seed=args.rseeds[0])(c) # not using
mid = Dense(units=args.units, activation='relu')(c)
out = Dense(1, activation='sigmoid')(mid)
model = Model(inputs=inputs, outputs=[out])
initial_weights = model.get_weights()
print('Done!\n\t--> Initialized model with {} layers and {} params.'.format(len(model.layers), model.count_params()))

# main loop to test each location
for loc in locs:
    skipped_folds = 0
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
    #dates_to_drop = [20180125, 20180806, 20150930, 20151211, 20161207, 20181123, 20180518, 20180113]
    #dates_to_drop += [20170418, 20180723, 20181024]
    dates_to_drop = []
    # df_loc contains observations for the current location; df_nloc is for neighboring or supplemental locations
    df_loc = gdelt.loc[(gdelt['Loc'] == loc) & (~gdelt['Date'].isin(dates_to_drop))].reset_index(drop=True)
    df_nloc = gdelt.loc[(gdelt['Loc'].isin(train_locs)) & (gdelt['Loc'] != loc) & (~gdelt['Date'].isin(dates_to_drop))].reset_index(drop=True)
    dates_to_exclude = sorted(df_loc['Date'].unique())[:delta]
    print('\t--> {} examples in location ({} events); {} in neighbor locations ({} events)'.format(len(df_loc), df_loc['label'].sum(), len(df_nloc), df_nloc['label'].sum()))

    folds = KFold(args.folds, shuffle=False).split(df_loc)
    # test each fold
    for i, fold in enumerate(folds):
        # preprocess data
        print('\nPreparing {} fold {} / {}...'.format(loc if not args.grouptest else '/'.join(locs), i + 1, args.folds))
        train_idx, test_idx = fold
        
        # if the training set doesn't contain enough events, we aggregate from neighbor locations
        if (df_loc.iloc[train_idx]['label'].sum() < args.eventthreshold) and not args.trainlocs and not args.grouptest:
            train_dates = df_loc.iloc[train_idx]['Date'].unique()
            train_locs = utils.agg_neighbors(gdelt.loc[gdelt['Date'].isin(train_dates)], loc, args.eventthreshold, normalize=True)
            df_nloc = gdelt.loc[(gdelt['Loc'].isin(train_locs)) & (gdelt['Loc'] != loc)]

        print('\n\tStacking training/testing data...')
        y_test = df_loc.iloc[test_idx].loc[~df_loc['Date'].isin(dates_to_exclude)]['label'].to_numpy()
        y_train = df_loc.iloc[train_idx].loc[~df_loc['Date'].isin(dates_to_exclude)]['label'].to_numpy()
        x_test = utils.stack(df_loc, delta, dates_to_exclude, idx=test_idx)
        x_train = utils.stack(df_loc, delta, dates_to_exclude, idx=train_idx)
        
        # if we have other locations to include in the training data (df_nloc) handle those cases below
        if not df_nloc.empty:
            # separate each additional location
            dfs_nloc = [df_nloc.loc[(df_nloc['Loc'] == loc)].reset_index(drop=True) for loc in df_nloc['Loc'].unique()]
            x_nloc = np.concatenate([utils.stack(df_t, delta, dates_to_exclude, idx=train_idx) for df_t in dfs_nloc])
            y_nloc = np.concatenate([df_t.iloc[train_idx].loc[~df_t['Date'].isin(dates_to_exclude)]['label'].to_numpy() for df_t in dfs_nloc])
            print('\tAppending {} examples from {} to training set...'.format(x_nloc.shape, train_locs))
            if args.droptestloc:
                x_train = x_nloc
                y_train = y_nloc
            else:
                x_train = np.concatenate([x_train, x_nloc])
                y_train = np.concatenate([y_train, y_nloc])
            if args.grouptest:
                x_nloc_test = np.concatenate([utils.stack(df_t, delta, dates_to_exclude, idx=test_idx) for df_t in dfs_nloc])
                y_nloc_test = np.concatenate([df_t.iloc[test_idx].loc[~df_t['Date'].isin(dates_to_exclude)]['label'].to_numpy() for df_t in dfs_nloc])
                x_test = np.concatenate([x_test, x_nloc_test])
                y_test = np.concatenate([y_test, y_nloc_test])
        
        # resample with SMOTE
        if args.resampling > 0:
            print('\nResampling from {} instances...'.format(len(x_train)))
            if args.resampling == 1:
                print('Resampling with SMOTE...')
                x_train = np.reshape(x_train, (x_train.shape[0], -1))
                x_train, y_train = SMOTE().fit_resample(x_train, y_train)
                x_train = np.reshape(x_train, (x_train.shape[0], -1, len(features)))
            elif args.resampling == 2:
                print('Resampling with random oversampling...')
                idx = np.arange(len(y_train)).reshape(-1, 1)
                samp_idx, y_train = RandomOverSampler().fit_resample(idx, y_train)
                x_train = x_train[samp_idx.flatten()]
            elif args.resampling == 3:
                print('Resampling with random over/undersampling...')
                idx = np.arange(len(y_train)).reshape(-1, 1)
                samp_idx, y_train = RandomOverSampler(sampling_strategy=0.5).fit_resample(idx, y_train)
                samp_idx, y_train = RandomUnderSampler().fit_resample(samp_idx, y_train)
                x_train = x_train[samp_idx.flatten()]
            print('Resampled {} training instances'.format(len(x_train)))
            
        # reorganize axes from samples/deltas/features to features/samples/deltas
        x_train = np.swapaxes(np.swapaxes(x_train, 1, 2), 0, 1)
        x_test = np.swapaxes(np.swapaxes(x_test, 1, 2), 0, 1)
        print('\tData Summary:')
        print('\t--> Training set shape: {}'.format(x_train.shape))
        print('\t--> Testing set shape {}'.format(x_test.shape))
        print('\t--> Events in training set: {} ({:.02f}%)'.format(np.sum(y_train), np.sum(y_train) * 100 / len(y_train)))
        print('\t--> Events in testing set: {} ({:.02f}%)'.format(np.sum(y_test), np.sum(y_test) * 100 / len(y_test)))
        # initialize model
        print('\tStarting training...')
        
        # train model and perform evaluation
        for seed in args.rseeds:
            # reset weights at each iteration
            model.set_weights(initial_weights)
            sgd = optimizers.SGD(lr=args.learningrate, decay=.000001, nesterov=True)
            model.compile(loss= SigmoidFocalCrossEntropy(),#balanced_cross_entropy(alpha=1-(np.sum(y_train) / len(y_train))) if not args.resampling else 'binary_crossentropy',
                          optimizer=sgd,
                          metrics=['accuracy', tf.keras.metrics.AUC()])
            
            history = model.fit([*x_train], y_train, epochs=n_epochs, batch_size=32, validation_data=([*x_test], y_test))

            output = model.predict([*x_test])
            preds = (np.argmax(output, axis=1))
            probs = output[:,-1]
            
            results['Loc'].append(loc if not args.grouptest else '/'.join(locs))
            results['Fold'].append(i)
            results['Seed'].append(seed)
            results['AUC'].append(roc_auc_score(y_test, probs) if y_test.sum() > 0 else np.nan)
            results['APR'].append(average_precision_score(y_test, probs) if y_test.sum() > 0 else np.nan)
            print('\t--> AUC: {:.4f}'.format(results['AUC'][-1]))
            print('\t--> APR: {:.4f}'.format(results['APR'][-1]))

    folds_processed = (i + 1) - skipped_folds
    print('\n*** Average AUC for {} ({} folds) : {:.4f} +- {:.4f}'.format(loc if not args.grouptest else '/'.join(locs), folds_processed, np.mean(results['AUC'][-folds_processed:]), np.std(results['AUC'][-folds_processed:])))
    print('\n*** Average APR for {} ({} folds) : {:.4f} +- {:.4f}'.format(loc if not args.grouptest else '/'.join(locs), folds_processed, np.mean(results['APR'][-folds_processed:]), np.std(results['APR'][-folds_processed:])))
    if args.grouptest:
        break
        
print('\nAll locations complete. Saving results to {}...'.format(out_dir + otf_name))
if not os.path.exists(out_dir):
    if not os.path.exists('results/'):
        os.mkdir('results/')
    os.mkdir(out_dir)
summary = pd.DataFrame.from_dict(results)
summary.to_csv(out_dir + otf_name, index=False)
print(summary.groupby(['Loc'])[['AUC','APR']].agg(['mean','std']))
#pd.concat(df_probs).sort_values(by=['Loc','Date','Seed']).to_csv(out_dir + otf_name[:-4] + '_preds.csv', index=False)
print('\nExecution complete!')
