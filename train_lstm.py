"""
Author: Steven Krieg (skrieg@nd.edu)
Dependencies: Python 3.7, tensorflow (2.0), sklearn (0.22), pandas (0.24), imblearn (0.6.2), tqdm (4.45.0)
Description:
    Trains an LSTM model on news features from GDELT to predict
    the occurrence of terrorist attacks.
    
    Input is gdelt features as a .csv file and gtd labels as a .csv file.
    
    Output is 2 .csv files: one for the 5-fold cross validation result, and one for daily predictions.
    
    *** the LSTM did not work as well as other models, so code is not as updated.
"""

import argparse
import os
import pandas as pd
import utils
import numpy as np
import tensorflow as tf
from datetime import datetime
from imblearn.keras import balanced_batch_generator
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold, KFold
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, concatenate, Dropout, LSTM
from tensorflow.keras import regularizers
from time import perf_counter
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('delta', type=int)
parser.add_argument('-l', '--locs', nargs='+', type=lambda x: x.upper(), default=['NY','CA','TX','FL','WA'])
parser.add_argument('-e', '--eventthreshold', type=int, default=10)
parser.add_argument('-p', '--predwindow', type=int, default=1)
parser.add_argument('-s', '--stratify', action='store_true')
parser.add_argument('-f', '--folds', type=int, default=5)
parser.add_argument('-b', '--nobalance', action='store_true')
parser.add_argument('-u', '--units', type=int, default=2000)
parser.add_argument('-i', '--epochs', type=int, default=100)
parser.add_argument('-r', '--rseeds', nargs='+', type=int, default=[777, 11767, 123818, 1, 200])
parser.add_argument('-q', '--debug', action='store_true')
args = parser.parse_args()
delta = args.delta
n_epochs = args.epochs
num_folds = args.folds

if args.debug:
    inf_gdelt = '/afs/crc.nd.edu/user/s/skrieg/Private/psicorp/dat/2019-11-01/gdelt-sample.csv'
else:
    inf_gdelt = '/afs/crc.nd.edu/user/s/skrieg/Private/psicorp/dat/2019-11-01/gdelt-features.csv'
inf_gtd = '/afs/crc.nd.edu/user/s/skrieg/Private/psicorp/gtd/gtd_relevant_events.csv'
out_dir = 'results/' + datetime.now().strftime('%Y%m%d') + '/'
otf_name = 'lstm_d{:02d}_{}ce.csv'.format(args.delta, 'smo' if args.nobalance else 'bal')
cols_to_drop = ['Date','Loc']
results = {'Loc':[], 'Fold': [], 'Seed': [], 'AUC': [], 'APR':[]}


def balanced_cross_entropy(alpha=.5):
    def balanced_cross_entropy_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        return -K.sum(alpha * K.log(pt_1)) - K.sum((1 - alpha) * K.log(1. - pt_0))
    return balanced_cross_entropy_fixed


print('Executing the following configuration.') 
for k, v in vars(args).items():
    print('\t-->{:>20} = {}'.format(k,v))

# initialize data
print()
print('-'*20)
print('Reading GDELT input from {}...'.format(inf_gdelt), end='')
gdelt = pd.read_csv(inf_gdelt)
print('Done!')
print('Reading GTD input from {}...'.format(inf_gtd), end='')
gtd = pd.read_csv(inf_gtd, encoding='ISO-8859-1', dtype=object, keep_default_na=False)
print('Done!\n')
print('-'*20)
print('Labelling features...')

utils.label_features(gdelt, gtd)
print('Done!\n')
print('-'*20)
print('Dropping states with zero events...', end='')
event_counts = gdelt[['Loc','label']].groupby(['Loc']).agg('sum')['label']
drop_states = event_counts[lambda x: x == 0].index
gdelt = gdelt.loc[~gdelt['Loc'].isin(drop_states)]
print('Dropped states {}.'.format(drop_states.to_numpy()))
locs = args.locs if args.locs else sorted(gdelt['Loc'].unique())
print('Testing on locs:')
for loc in locs: 
    print('\t--> {} ({} events)'.format(loc, event_counts.at[loc]))

# normalize df
print('Normalizing input data...', end='')
features = [c for c in gdelt.columns if c not in cols_to_drop and c != 'label']
gdelt[features] = MinMaxScaler().fit_transform(gdelt[features])
print('Done!')
# initialize model architecture
print('Initializing model architecture...', end='')
model = Sequential()
model.add(LSTM(units=args.units, input_shape=(args.delta, len(features)), activation='tanh', recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
initial_weights = model.get_weights()
print('Done!\n\t--> Initialized model with {} layers and {} params.'.format(len(model.layers), model.count_params()))

# main loop to test each location
for loc in locs:
    skipped_folds = 0
    print('\n{}\nGenerating model for {}...'.format('-' * 20, loc))
    train_locs = utils.agg_neighbors(gdelt, loc, args.eventthreshold, normalize=False)
    #dates_to_drop = [20180125, 20180806, 20150930, 20151211, 20161207, 20181123, 20180518, 20180113]
    #dates_to_drop += [20170418, 20180723, 20181024]
    dates_to_drop = []
    df_loc = gdelt.loc[(gdelt['Loc'] == loc) & (~gdelt['Date'].isin(dates_to_drop))].reset_index(drop=True)
    df_nloc = gdelt.loc[(gdelt['Loc'].isin(train_locs)) & (gdelt['Loc'] != loc)].reset_index(drop=True)
    dates_to_exclude = sorted(df_loc['Date'].unique())[:delta]
    print('\t--> {} examples in location ({} events); {} in neighbor locations ({} events)'.format(len(df_loc), df_loc['label'].sum(), len(df_nloc), df_nloc['label'].sum()))
    print('\tStacking neighbor data...')
    if not df_nloc.empty:
        x_nloc = utils.stack(df_nloc, delta, dates_to_exclude)
        y_nloc = df_nloc.loc[~df_nloc['Date'].isin(dates_to_exclude)]['label'].to_numpy()

    for seed in args.rseeds:
        if args.stratify:
            folds = StratifiedKFold(args.folds, shuffle=True, random_state=seed).split(df_loc, df_loc['label'])
            #folds = StratifiedKFold(df_loc['label'].sum(), shuffle=True, random_state=seed).split(df_loc, df_loc['label'])
        else:
            folds = KFold(args.folds, shuffle=False).split(df_loc)

        # test each fold
        for i, fold in enumerate(folds):
            # preprocess data
            print('\nPreparing {} fold {}...'.format(loc, i + 1))
            train_idx, test_idx = fold
            print('\n\tStacking training/testing data...')
            y_test = df_loc.iloc[test_idx].loc[~df_loc['Date'].isin(dates_to_exclude)]['label'].to_numpy()
            if np.sum(y_test) < 1:
                print('\t--> No events in testing set. Skipping fold.')
                skipped_folds += 1
                continue
            y_train = df_loc.iloc[train_idx].loc[~df_loc['Date'].isin(dates_to_exclude)]['label'].to_numpy()
            #y_train = df_loc.loc[~df_loc['Date'].isin(dates_to_exclude)]['label'].to_numpy()
            if not df_nloc.empty: y_train = np.concatenate([y_train, y_nloc])
            if np.sum(y_train) < 6:
                print('\t--> Not enough events in training set. Skipping fold.')
                skipped_folds += 1
                continue
            x_test = utils.stack(df_loc, delta, dates_to_exclude, idx=test_idx)
            # reorganize axes from samples/deltas/features to features/samples/deltas
            # x_test = np.swapaxes(np.swapaxes(x_test, 1, 2), 0, 1)
            x_train = utils.stack(df_loc, delta, dates_to_exclude, idx=train_idx)
            #x_train = utils.stack(df_loc, delta, dates_to_exclude)
            if not df_nloc.empty: x_train = np.concatenate([x_train, x_nloc])
            # reorganize axes from samples/deltas/features to features/samples/deltas
            if args.nobalance:
                x_train = np.reshape(x_train, (x_train.shape[0], -1))
                x_train, y_train = SMOTE().fit_resample(x_train, y_train)
                x_train = np.reshape(x_train, (x_train.shape[0], -1, len(features)))
            # x_train = np.swapaxes(np.swapaxes(x_train, 1, 2), 0, 1)
            print('\tData Summary:')
            print('\t--> Training set shape: {}'.format(x_train.shape))
            print('\t--> Testing set shape {}'.format(x_test.shape))
            print('\t--> Events in training set: {} ({:.02f}%)'.format(np.sum(y_train), np.sum(y_train) * 100 / len(y_train)))
            print('\t--> Events in testing set: {} ({:.02f}%)'.format(np.sum(y_test), np.sum(y_test) * 100 / len(y_test)))
            # initialize model
            print('\tStarting training...')
            
            # reset weights
            model.set_weights(initial_weights)
            # sgd = optimizers.SGD(lr=.001, decay=.0001, nesterov=True)
            model.compile(loss='binary_crossentropy', #balanced_cross_entropy(alpha=1-(np.sum(y_train) / len(y_train))),
                          optimizer=optimizers.Adam(learning_rate=0.0001),
                          metrics=['accuracy', tf.keras.metrics.AUC()])
            
            #generator, steps = balanced_batch_generator([*x_train], y_train, batch_size=128)
            #history = model.fit_generator(generator=generator, steps_per_epoch=steps,epochs=n_epochs, validation_data=([*x_test], y_test))
            history = model.fit(x_train, y_train, epochs=n_epochs, batch_size=64, validation_data=(x_test, y_test)) #, class_weight=compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train))

            output = model.predict(x_test)
            preds = (np.argmax(output, axis=1)) 
            probs = output[:,-1]
            
            results['Loc'].append(loc)
            results['Fold'].append(i)
            results['Seed'].append(seed)
            results['AUC'].append(roc_auc_score(y_test, probs))
            results['APR'].append(average_precision_score(y_test, probs))
            print('\t--> AUC: {:.4f}'.format(results['AUC'][-1]))
            print('\t--> APR: {:.4f}'.format(results['APR'][-1]))

        folds_processed = (i + 1) - skipped_folds
        print('\n*** Average AUC for {} ({} folds) : {:.4f} +- {:.4f}'.format(loc, folds_processed, np.mean(results['AUC'][-folds_processed:]), np.std(results['AUC'][-folds_processed:])))
        print('\n*** Average APR for {} ({} folds) : {:.4f} +- {:.4f}'.format(loc, folds_processed, np.mean(results['APR'][-folds_processed:]), np.std(results['APR'][-folds_processed:])))

print('\nAll locations complete. Saving results to {}...'.format(out_dir + otf_name))
if not os.path.exists(out_dir):
    if not os.path.exists('results/'):
        os.mkdir('results/')
    os.mkdir(out_dir)
summary = pd.DataFrame.from_dict(results)
summary.to_csv(out_dir + otf_name, index=False)
print(summary)
print(summary.groupby(['Loc','Fold'])[['AUC','APR']].agg(['mean','std']))
print(summary.groupby(['Loc'])[['AUC','APR']].agg(['mean','std']))
print('\nExecution complete!')
