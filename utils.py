# utils.py
# author: skrieg
# provides a set of utilities for use in all models

import argparse
import pandas as pd
import multiprocessing as mp
import numpy as np
from itertools import product
from scipy.spatial.distance import pdist
from scipy.stats import ks_2samp, ttest_ind
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

# for tqdm
bar_fmt='{l_bar:>13}{bar:20}{r_bar}{bar:-20b}'

# adds datetime columns to gdelt
def add_dates(gdelt):
    dtime = pd.to_datetime(gdelt['Date'], format='%Y%m%d')
    gdelt['DayOfWeek'] = dtime.dt.dayofweek
    gdelt['Year'] = dtime.dt.year
    gdelt['Month'] = dtime.dt.month

# adjusts training/test folds for observation windows to ensure each split has integrity
def adjust_folds(folds, delta):
    new_folds = []
    for fold in folds:
        train_idx, test_idx = fold
        test_min = np.min(test_idx)
        # because observation windows use a moving average we have to exclude from the training set
        # any examples that are included in a moving average calculation for a testing example
        # this can result in the first testing set being disproportionate, so we address that as well
        to_shift = train_idx[(train_idx >= (test_min - delta)) & (train_idx < test_min)]
        if to_shift.size:
            new_folds.append((train_idx[~np.isin(train_idx, to_shift)], np.concatenate([to_shift, test_idx])))
        else:
            new_folds.append((train_idx, test_idx))
    return new_folds

# hierarchical clustering method for grouping similar states based on news features
# event_threshold (int) : keep clustering until the data has this many events
# normalize (boolean): whether to normalize (standard score) news features before clustering
def agg_neighbors(df, loc, event_threshold, normalize=True):
    loc_group = [loc]
    event_count = df.loc[df['Loc'].isin(loc_group)]['label'].sum()
    if event_count < event_threshold:
        print('\t{} has only {} events; aggregating neighbor locations...'.format(loc, event_count))
        candidates = df.drop(['Date','label'], axis=1).groupby('Loc').agg('mean')
        if normalize: 
            candidates[candidates.columns] = MinMaxScaler().fit_transform(candidates[candidates.columns])
        distances = pd.Series({c: pdist(candidates.loc[[loc, c]].to_numpy(), metric='euclidean')[0] for c in candidates.index if c != loc})
        while event_count < event_threshold:
            choice = distances.idxmin()
            count = df.loc[df['Loc'] == choice]['label'].sum()
            print('\t--> Aggregating {} (distance {} with event count {})'.format(choice, distances.at[choice], count))
            event_count += count
            loc_group.append(choice)
            distances.drop(choice, inplace=True)
        print('\tAggregated locations: {}'.format(loc_group))
    return loc_group[1:]

# used for prediction windows method 2: aggregating dates_to_exclude
# e.g. Jan 1, 2, 3 with prediction window of 3 would become a single observation Jan 1-3
def agg_dates(df, features, window):
    irange = list(range(window + ((len(df['Date'].unique()) - 1) % window), len(df['Date'].unique()) , window))
    df_agg = df.sort_values(['Loc','Date']).groupby('Loc')[features].rolling(window).mean()
    df_agg['Date'] = df.sort_values(['Loc','Date']).groupby('Loc')['Date'].rolling(window).max().fillna(0).astype(int)
    df_agg['label'] = df.sort_values(['Loc','Date']).groupby('Loc')['label'].rolling(window).max().fillna(0).astype(int)
    df_agg = df_agg.reset_index(level=0).groupby('Loc').nth(irange).set_index('Date', append=True).reset_index()
    return df_agg

# compute the observation windows for each feature by maximizing the kolmogorov-smirnov (K-S) distance between the classes
# n_workers (int): number of processes for the multiprocessing module to create (i.e. # cpus)
# delta_range (list of int): the list of possible values
# delta_map (dict of string:int): apply the same windows to the test set that were computed on the training set
# normalize (boolean): whether to normalize (standard score) the features before computing deltas
# use_tonesum (boolean): not used
def compute_deltas(df_in, n_workers, delta_range=[], delta_map=None, normalize=False, use_tonesum=False):
    if delta_map is None:
        delta_map = {}
    df = df_in.copy(deep=True)
    results = {'Delta': [], 'Feature': [], 'Distance': []}
    inq = mp.Queue()
    outq = mp.Queue()
    outputs = []
    # for debugging only
    # df.drop(df.columns[2:-7], axis=1, inplace=True)
    features = list(df.columns[2:-1])
    n_locs = len(df['Loc'].unique())
    
    # reverse-engineers toneavg to be normalized by count
    # does not improve performance, so we don't use it
    if use_tonesum:
        for col in [c for c in df.columns if '_toneavg' in c]:
            df[col] = (df[col] * df[col.replace('_toneavg', '_count')]).astype(int)
    
    for feature in features:
        inq.put(df[['Loc', feature, df.columns[-1]]])

    workers = [mp.Process(target=compute_deltas_worker, args=(inq, outq, delta_range, delta_map)) for n in range(n_workers)]
    for w in workers:
        inq.put(pd.DataFrame())   # termination criteria
        w.start()
    
    for i in tqdm(range(len(features)), mininterval=1, bar_format=bar_fmt):
        df_out, best_delta, max_dist = outq.get()
        outputs.append(df_out)
        delta_map[df_out.columns[1]] = (best_delta, max_dist)

    # same as above; not used
    if use_tonesum:
        print('\tRefreshing toneavg values...', end='')
        workers = [mp.Process(target=refresh_toneavg_worker, args=(inq, outq, df_in[[c for c in features if '_count' in c]])) for n in range(n_workers)]
        jobs = [out for out in outputs if '_toneavg' in out.columns[1]]
        outputs = [out for out in outputs if '_toneavg' not in out.columns[1]]
        [inq.put((out, delta_map[out.columns[1]][0])) for out in jobs]
        for w in workers:
            inq.put((pd.DataFrame(), 0))
            w.start()
        for i in range(len(jobs)):
            outputs.append(outq.get())
        print('Done!')
    
    # truncate the first delta observations from the beginning of the data (for each location)
    max_delta = int(max([len(df)-len(series) for series in outputs]) / n_locs)
    df_out = df[['Loc','Date',df.columns[-1]]].groupby('Loc', as_index=False).apply(lambda x: x.iloc[max_delta:])
    outputs = [output.groupby('Loc', as_index=False).apply(lambda x: x.iloc[max_delta-int((len(df)-len(output))/n_locs):]).drop('Loc', axis=1) for output in outputs]
    df_out = pd.concat([df_out] + outputs, axis=1)

    if normalize: normalize_features(df_out)
    # re-order colulmns on return
    return (df_out[['Loc','Date'] + sorted(df_out.columns[3:]) + ['label']].reset_index(drop=True), delta_map)

# worker function for computing deltas
# if delta_map is not provided (training set) we choose the best delta from options in delta_range
# if delta_map is provided (testing set) we ignore delta_range and compute each feature using delta_map
def compute_deltas_worker(inq, outq, delta_range, delta_map):
    df_in = inq.get() # df_in contains a single feature
    
    while not df_in.empty: # empty df is the termination signal
        assert len(df_in.columns) == 3
        if len(df_in[df_in.columns[1]].unique()) == 1: # if a feature has only one value, e.g. 0.0
            outq.put((df_in.iloc[:,:2], 1, 0))
        else:
            feature = df_in.columns[1]
            # we want to compute the minimum pval and argmin
            best_delta, min_pval = 0, np.inf
            df_out = pd.DataFrame()
            locs = df_in['Loc'].unique()
            
            if feature in delta_map:
                deltas = [delta_map[feature][0]]
            else:
                deltas = delta_range
            
            pvals = {}
            for delta in deltas:
                df = pd.concat([compute_feature(df_in.loc[df_in['Loc'] == loc], delta) for loc in locs])
                pval = compute_distance(df.loc[df[df.columns[-1]] == 0][feature], df.loc[df[df.columns[-1]] == 1][feature])
                pvals[delta] = pval
            # if there is only one delta (i.e. on the test set) just return that one
            if len(deltas) == 1:
                df_out = df
                best_delta = delta
                min_pval = pval
            else:
                s = pd.Series(pvals)
                min_pval = s.max()
                best_delta = s.idxmax()
                df_out = pd.concat([compute_feature(df_in.loc[df_in['Loc'] == loc], best_delta) for loc in locs])
            
            # return the output
            outq.put((df_out.iloc[:,:2], best_delta, min_pval))
        df_in = inq.get()

# compute the k-s distance between two samples
def compute_distance(d1, d2):
    if d2.sum() > 0 and d1.sum() > 0:
        return ks_2samp(d1, d2)[0]
        #return abs(ttest_ind(d1, d2, equal_var=False)[0])
    else:
        return 0

# compute the moving average of a single feature based on a value of delta
# (this function is called by compute_deltas_worker)
def compute_feature(df_in, delta, trim=0):
    # assume df_in columns are ordered as [loc, feature, label]
    assert len(df_in.columns) == 3
    df = df_in.copy(deep=True)
    df[df.columns[1]] = df_in[df_in.columns[1]].astype(float).rolling(delta, win_type=None).mean().shift(1)
    return df.iloc[delta+trim:]

# not used
def decompose_features(df):
    for col in tqdm(list(df.columns)):
        decomp = seasonal_decompose(df[col], period=7, two_sided=False, extrapolate_trend=1)
        df.loc[:,col] = decomp.trend + decomp.resid
    return df

# labels the gdelt features using events in the gtd
# gdelt and gtd are both dataframes
# window (int): propagate labels backwards this number of days
# drop (boolean): whether to drop melee/unarmed terrorist attacks
def label_features(gdelt, gtd, window=1, drop=False):
    gdelt_dates = set(gdelt['Date'])
    loc_map = {'Alabama':'AL','Alaska':'AK','Arizona':'AZ','Arkansas':'AR',
               'California':'CA','Colorado':'CO','Connecticut':'CT','Delaware':'DE',
               'District of Columbia':'DC','Florida':'FL','Georgia':'GA',
               'Hawaii':'HI','Idaho':'ID','Illinois':'IL','Indiana':'IN','Iowa':'IA',
               'Kansas':'KS','Kentucky':'KY','Louisiana':'LA','Maine':'ME',
               'Maryland':'MD','Massachusetts':'MA','Michigan':'MI','Minnesota':'MN',
               'Mississippi':'MS','Missouri':'MO','Montana':'MT','Nebraska':'NE',
               'Nevada':'NV','New Hampshire':'NH','New Jersey':'NJ','New Mexico':'NM',
               'New York':'NY','North Carolina':'NC','North Dakota':'ND','Ohio':'OH',
               'Oklahoma':'OK','Oregon':'OR','Pennsylvania':'PA','Rhode Island':'RI',
               'South Carolina':'SC','South Dakota':'SD','Tennessee':'TN','Texas':'TX',
               'Utah':'UT','Vermont':'VT','Virginia':'VA','Washington':'WA',
               'West Virginia':'WV','Wisconsin':'WI','Wyoming':'WY' }
    events = []
    if drop:
         gtd = gtd.loc[(gtd['weaptype1_txt'] != 'Melee') & (gtd['attacktype1_txt'] != 'Unarmed Assault')]
    
    for i, row in gtd.loc[(gtd['country_txt'] == 'United States')].iterrows():
        date = int('{:04d}{:02d}{:02d}'.format(int(row['iyear']), int(row['imonth']), int(row['iday'])))
        if date in gdelt_dates and row['provstate'] in loc_map:
            events.append((date, loc_map[row['provstate']]))
        else:
            print('--> Date/loc in GTD not found in GDELT data: {}'.format(str((date, row['provstate']))))
            
    events = set(events)
    print('Labelling {} unique date/location combinations with window {}...'.format(len(events), window), end='')
    gdelt['label'] = gdelt.apply(lambda row: 1 if (row['Date'], row['Loc']) in events else 0, axis=1)
    if window > 1:
        gdelt['label'] = gdelt.groupby('Loc')['label'].apply(lambda x: x.iloc[::-1].rolling(window=window,min_periods=1).max().astype(int).iloc[::-1])
    gdelt.sort_values(['Loc','Date'], inplace=True)
    print('Labeled {} events.'.format(gdelt['label'].sum()))

# normalize features via standard score (z-score) normalization x = (x - x.mean) / x.stdev
def normalize_features(df):
    for col in df.columns[2:-1]:
        try:
            frange = (-1,1) if '_toneavg' in col else (0,1)
            df[col] = MinMaxScaler(feature_range=frange).fit_transform(df[col].astype('float64').values.reshape(-1, 1))
        except:
            print(col)
            print(list(df[col]))

# test driver function - not used
def preprocess(inf_gdelt, inf_gtd, delta_range=[1,3,5,7,14,30,60,90,120,150,180], n_workers=None):
    if not n_workers: n_workers = mp.cpu_count()
 
    print('Reading GDELT input from {}...'.format(inf_gdelt))
    gdelt = pd.read_csv(inf_gdelt)
   
    print('Reading GTD input from {}...'.format(inf_gtd))
    gtd = pd.read_csv(inf_gtd, encoding='ISO-8859-1', dtype=object, keep_default_na=False)
    
    print('Labelling features...')
    label_features(gdelt, gtd)
    
    return compute_deltas(gdelt, n_workers, delta_range=delta_range)
    

# used to recompute toneavgs from tonesums - not used
def refresh_toneavg_worker(inq, outq, df_in):
    out, tone_delta = inq.get()
    while not out.empty:
        col = out.columns[1]
        count_col = col.replace('_toneavg','_count')
        count_sums = df_in[count_col].copy().rolling(tone_delta).sum().shift(1)
        out[col] = out[col].div(count_sums.iloc[tone_delta:]).fillna(0.0)
        outq.put(out)
        out, tone_delta = inq.get()

# used to propagate labels backwards for larger prediction windows
def shift_window(y, window):
    return pd.Series(y).rolling(window=window, center=True, min_periods=1).max().astype(int).to_numpy()

# used by all neural network models
# stacks the daily observations into a 3d numpy matrix
# i.e. df comes in as a matrix: nobservations x nfeatures
# and we return a matrix that is nobservations x delta x nfeatures
# dates_to_exclude (list of int): any dates we want to drop
# idx (np array of int): the idx we want to keep, i.e. for train/test split
def stack(df, delta, dates_to_exclude, idx=np.array([])):
    if idx.size:
        return np.stack([df.iloc[i-delta:i].drop(['Date','Loc','label'], axis=1).values 
                        for i in tqdm(df.iloc[idx].loc[~df['Date'].isin(dates_to_exclude)].index, bar_format=bar_fmt)], axis=0)
    else:
        return np.stack([df.iloc[i-delta:i].drop(['Date','Loc','label'], axis=1).values 
                        for i in tqdm(df.loc[~df['Date'].isin(dates_to_exclude)].index, bar_format=bar_fmt)], axis=0)

# truncates the first delta observations of the feature matrix
def truncate_stack(x, max_delta, delta_map):
    # re align axes to observations / features / deltas
    x = np.swapaxes(x, 0, 1)
    x = np.reshape(x, (x.shape[0], -1))
    masks = []
    for i, (feature, (delta, pval)) in enumerate(delta_map.items()):
        masks.append(np.ones(delta, dtype=bool))
        masks.append(np.zeros(max_delta - delta, dtype=bool))
    mask = np.concatenate(masks, axis=None)
    print('Shape of original (flattened) data:', x.shape)
    x = x[:,mask]
    print('New shape:', x.shape)
    return x

# test driver
if __name__ == '__main__':
    inf_gdelt = '/afs/crc.nd.edu/user/s/skrieg/Private/psicorp/dat/2019-11-01/gdelt-sample.csv'
    inf_gtd = '/afs/crc.nd.edu/user/s/skrieg/Private/psicorp/gtd/gtd_relevant_events.csv'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--workers', type=int, default=1)
    args = parser.parse_args()
     
    df = preprocess(inf_gdelt, inf_gtd, n_workers=args.workers)
    df.to_csv('test.csv', index=False)