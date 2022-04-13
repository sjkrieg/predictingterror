"""
Author: Steven Krieg (skrieg@nd.edu)
Dependencies: Python 3.7, Pandas 0.24.2
Description:
    Data preprocessing pipeline for the GDELT database. 
    Input is raw files from the GKG and events tables.
    Output is a combination of counts and average tones
    (semantic scores) for a list of themes and CAMEO
    event codes.
    Themes are extracted from the GKG, and CAMEO codes 
    from the events table.
    References are then grouped by location.
    
themes and cameo dictionaries are structured as follows:
    -> date
        -> location
            -> theme / cameo
                -> 'count'
                    -> (int)
                -> 'tone_sum'
                    -> (float)
"""

# User specified parameters:
#   num_workers: leave as None to use multiprocessing.cpu_count()

default_start_date = '20190701'
default_end_date = '20190702'
default_num_workers = None  # uses mp.cpu_count()
default_target_dir = 'gdelt_exports/'

gdelt_file_url = 'http://data.gdeltproject.org/gdeltv2/masterfilelist.txt'
gkg_search_str = '.gkg.csv.zip'
event_search_str = '.export.CSV.zip'
delete_downloaded_files = True
inf_cameo_list = 'ref/cameo_eventbasecodes.csv'
inf_theme_list = 'ref/themes.txt'
cameo_column = 'EventBaseCode'

# imports & other definitions
from urllib.request import urlopen, urlretrieve
import os
import zipfile
from time import perf_counter, gmtime, strftime
import pandas as pd
import multiprocessing as mp
import datetime
import argparse
import platform  

TERMINATE_CMD = '__DONE__'
SUCCESS_MSG = '__SUCCESS__'
FAILED_MSG = '__FAILED__'

loc_set = set(['AK','AL','AZ','AR','CA','CO','CT','DC','DE','FL','GA',
                   'HI','ID','IL','IN','IA','KS','KY','LA','ME','MD','MA',
                   'MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY',
                   'NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX',
                   'UT','VT','VA','WA','WV','WI','WY'])

gkg_cols = ['GKGRECORDID','DATE','SourceCollectionIdentifier','SourceCommonName',
            'DocumentIdentifier','Counts','V2Counts','Themes','V2Themes',
            'Locations','V2Locations','Persons','V2Persons','Organizations',
            'V2Organizations','V2Tone','Dates','GCAM','SharingImage',
            'RelatedImages','SocialImageEmbeds','SocialVideoEmbeds','Quotations',
            'AllNames','Amounts','TranslationInfo','Extras']

event_cols = ['GLOBALEVENTID','SQLDATE','MonthYear','Year','FractionDate',
              'Actor1Code','Actor1Name','Actor1CountryCode','Actor1KnownGroupCode',
              'Actor1EthnicCode','Actor1Religion1Code','Actor1Religion2Code',
              'Actor1Type1Code','Actor1Type2Code','Actor1Type3Code','Actor2Code',
              'Actor2Name','Actor2CountryCode','Actor2KnownGroupCode',
              'Actor2EthnicCode','Actor2Religion1Code','Actor2Religion2Code',
              'Actor2Type1Code','Actor2Type2Code','Actor2Type3Code','IsRootEvent',
              'EventCode','EventBaseCode','EventRootCode','QuadClass',
              'GoldsteinScale','NumMentions','NumSources','NumArticles','AvgTone',
              'Actor1Geo_Type','Actor1Geo_FullName','Actor1Geo_CountryCode',
              'Actor1Geo_ADM1Code','Actor1Geo_ADM2Code','Actor1Geo_Lat',
              'Actor1Geo_Long','Actor1Geo_FeatureID','Actor2Geo_Type',
              'Actor2Geo_FullName','Actor2Geo_CountryCode','Actor2Geo_ADM1Code',
              'Actor2Geo_ADM2Code','Actor2Geo_Lat','Actor2Geo_Long','Actor2Geo_FeatureID',
              'ActionGeo_Type','ActionGeo_FullName','ActionGeo_CountryCode',
              'ActionGeo_ADM1Code','ActionGeo_ADM2Code','ActionGeo_Lat',
              'ActionGeo_Long','ActionGeo_FeatureID','DATEADDED','SOURCEURL']
    
event_cols_to_use = ['SQLDATE','ActionGeo_ADM1Code',cameo_column,'NumMentions','AvgTone']

# define worker function
def gdelt_worker(date_list, theme_map, cameo_set, feature_set, inq, outq, msgq, target_dir):
    scores = {d: {loc: {f: {'count': 0, 'tone_sum': 0.0} for f in feature_set} for loc in loc_set} for d in date_list}
    url = inq.get()
    while url != TERMINATE_CMD:
        # large try block to ensure program does not crash for a bad file
        try:
            file = url.split('/')[-1]
    
            # check if the CSV already exists from a previous download
            if os.path.exists(target_dir + file[:-4]):
                file = file[:-4]
            else:
                # download the remote file
                urlretrieve(url, filename=target_dir + file)
            
                # unzip - should handle multiple files per zip
                with zipfile.ZipFile(target_dir + file, 'r') as zipped:
                    csv_names = zipped.namelist()
                    zipped.extractall(target_dir)
                # don't need to keep the zip
                os.remove(target_dir + file)
                file = csv_names[0]
            
            raw_size = os.path.getsize(target_dir + file)
            
            # if a gkg file
            if url.endswith(gkg_search_str):
                # keep_default_na=False prevents pandas from reading empty values as float, which causes problems for the split function
                df = pd.read_csv(target_dir + file, delimiter='\t', names=gkg_cols, keep_default_na=False, encoding='ISO-8859-1')
                #df.to_csv('gkg_sample.csv')
                for i, row in df.iterrows():
                    if row['V2Locations']:
                        cur_locs = row['V2Locations'].strip().split(';')
                        for theme in row['V2Themes'].strip().split(';'):
                            theme = theme.split(',')[0].strip()
                            if theme in theme_map:
                                theme = theme_map[theme]
                                date = str(row['DATE'])[:8]
                                for loc in cur_locs:
                                    loc_splits = loc.split('#')
                                    if loc_splits[0] == '2' or loc_splits[0] == '3':
                                        loc = loc_splits[3][2:]
                                        
                                        if loc in loc_set:
                                            scores[date][loc][theme]['count'] += 1
                                            scores[date][loc][theme]['tone_sum'] += float(row['V2Tone'].split(',')[0])
        
            # if an event file
            elif url.endswith(event_search_str):
                df = pd.read_csv(target_dir + file, delimiter='\t', names=event_cols, dtype={'EventBaseCode': str}, keep_default_na=False, encoding='ISO-8859-1')[event_cols_to_use]
                for i, row in df.iterrows():
                    loc = row['ActionGeo_ADM1Code']
                    if loc.startswith('US') and loc[2:] in loc_set and row[cameo_column] in cameo_set:
                        try:
                            date = str(row['SQLDATE'])
                            loc = loc[2:]
                            event = row[cameo_column]
                            scores[date][loc][event]['count'] += row['NumMentions']
                            scores[date][loc][event]['tone_sum'] += (row['AvgTone'] * row['NumMentions'])
                        except:
                            # exception is most likely a date that is invalid or out of range
                            pass
                        
            msgq.put((SUCCESS_MSG, file, raw_size))
        # end try
                    
        # any exception related to processing the input file would be caught here
        except Exception as e:
            """
            import traceback
            print(traceback.print_exc())
            raise SystemExit
            """
            msgq.put((FAILED_MSG, f'{file}: {e}'))
        
        if delete_downloaded_files:
            try:
                os.remove(target_dir + file)
            except Exception:
                pass
        
        # now download the corresponding EVENT file
        url = inq.get()
    
    # end while
        
    # tell the driver we are done and send the total number of date mentions for reporting
    msgq.put((TERMINATE_CMD,))
    outq.put(scores)
    # end of worker function
    

if __name__ == '__main__':
    # initializations
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('startdate')
        parser.add_argument('enddate')
        parser.add_argument('numworkers', type=int)
        parser.add_argument('--outdir', '-o', default=default_target_dir)
        args = parser.parse_args()
        start_date = args.startdate
        end_date = args.enddate 
        num_workers = args.numworkers
    except:
        print('Command line arguments not provided; using default values.')
        num_workers = default_num_workers
        start_date = default_start_date
        end_date = default_end_date
        
    target_dir = args.outdir
    theme_map = {theme: theme for theme in pd.read_csv(inf_theme_list, delimiter='\t')['Name']}
    cameo_set = set(pd.read_csv(inf_cameo_list, delimiter=',', dtype={'EventBaseCode': str})['EventBaseCode'])
    feature_set = cameo_set.union(set(theme_map.values()))
    features_ordered = sorted(theme_map.values()) + sorted(cameo_set)
    
    start_datetime = datetime.datetime.strptime(start_date, '%Y%m%d')
    try:
        end_datetime = datetime.datetime.strptime(end_date, '%Y%m%d')
    except:
        # to handle months with different numbers of days
        end_datetime = datetime.datetime.strptime(end_date[:6] + '01', '%Y%m%d')
        end_datetime = end_datetime.replace(month = end_datetime.month + 1) - datetime.timedelta(days=1)
    
    if start_date != end_date:
        otf_name = start_datetime.strftime('%Y%m%d') + '-' + end_datetime.strftime('%Y%m%d') + '.csv'
    else:
        otf_name = start_datetime.strftime('%Y%m%d') + '.csv'
        
    date_list = [(start_datetime + datetime.timedelta(days=i)).strftime('%Y%m%d') for i in range((end_datetime - start_datetime).days + 1)]
    
    scores = {}
    to_download = []
    download_size = 0
    raw_size = 0
    progress_interval = 20
    done_count = 0 # for tracking worker processes
    success_count = 0
    failed_count = 0
    total_ref_count = 0
    failed_msgs = []
    start = perf_counter()
    if not num_workers: num_workers = mp.cpu_count()
    inq = mp.Queue() # for worker queue to receive tasks
    outq = mp.Queue() # for sending results from worker to driver
    msgq = mp.Queue()
    
    # get a list of all files to download
    print(f'Querying {gdelt_file_url} for gdelt files from {start_datetime.strftime("%Y-%m-%d")} to {end_datetime.strftime("%Y-%m-%d")}...')
    for i, line in enumerate(urlopen(gdelt_file_url)):
        line = line.decode('utf-8').split(' ')
        if len(line) == 3:
            size = line[0]
            url = line[2].strip()
            if url.endswith(gkg_search_str) or url.endswith(event_search_str):
                file_date = url.split('/')[-1][:len(start_date)]
                if file_date >= start_date and file_date <= end_date:
                    to_download.append(url)
                    download_size += int(size)
    print(f'Found {len(to_download)} files. Total download size for compressed files is {download_size >> 20}MB.')

    if len(to_download):
        # check to make sure target dir exists
        if not (os.path.exists(target_dir) and os.path.isdir(target_dir)):
            os.mkdir(target_dir)
            print(f'Created target directory {target_dir}.')
        
        # populate the worker queue
        print(f'Using {num_workers} workers to download {len(to_download)} files to target directory {target_dir}...')
        for i, url in enumerate(to_download):
            inq.put(url)
        
        # add a terminate command to the end of queue so workers know when to stop
        # this seems more robust than having them check if the queue is empty
        for i in range(num_workers):
            inq.put(TERMINATE_CMD)
        
        # start the worker processes
        workers = [mp.Process(target=gdelt_worker, args=(date_list, theme_map, cameo_set, feature_set, inq, outq, msgq, target_dir)) for n in range(num_workers)]
        [w.start() for w in workers]
        
        # for debugging, since we can't print from separate processes
        #gdelt_worker(start_date, end_date, theme_map, inq, outq, msgq, target_dir)
        
        while done_count < num_workers:
            # we only pass tuple of str to msgq
            # try block is necessary to prevent program from crashing on msgq.get() timeout
            # I like the timeout; it allows the progress bar to update even without getting a message
            try:
                msg = msgq.get(timeout=3)
                
                if msg[0] == SUCCESS_MSG:
                    success_count += 1
                    raw_size += msg[2]
                elif msg[0] == FAILED_MSG:
                    failed_count += 1
                    failed_msgs.append(msg[1])
                elif msg[0] == TERMINATE_CMD:
                    done_count += 1
                    
            except:
                pass
            
            finally:
                # update the progress bar
                progress_pct = int((100 * (success_count + failed_count)) / len(to_download))
                progress_bar = int(progress_pct * progress_interval / 100)
                print('\r(' + ('-' * progress_bar) + '>' + (' ' * (progress_interval - progress_bar)) + f') {progress_pct}% ({success_count:,d} files successfully extracted ({raw_size >> 20:,d}MB) and {failed_count} failed in {strftime("%H:%M:%S", gmtime(perf_counter() - start))})', end='')        
        
        print('\nDownloads complete. Combining counts...', end='')
        results = [outq.get() for w in workers]
        scores = results.pop()
        for result in results:
            for date, t1 in result.items():
                for loc, t2 in t1.items():
                    for theme, vals in t2.items():
                        scores[date][loc][theme]['count'] += vals['count']
                        scores[date][loc][theme]['tone_sum'] += vals['tone_sum']
                        
        print(f'Done!\nWriting results to {target_dir + otf_name}...', end='')
        with open(target_dir + otf_name, 'w+') as otf:
            otf.write('Date,Loc,' + ','.join([('theme_' + str(f) + '_count,theme_' + str(f) + '_toneavg') if not f[:3].isdigit() else ('cameo_' + str(f) + '_count,cameo_' + str(f) + '_toneavg') for f in features_ordered]) + '\n')
            for date, t1 in scores.items():
                for loc, vals in t1.items():
                    otf.write('%s,%s' % (date, loc))
                    for feature in features_ordered:
                        count = vals[feature]['count']
                        otf.write(',%d,%f' % (count, vals[feature]['tone_sum'] / count if count else 0.0))
                    otf.write('\n')

        print('Done!\n')
        print('*' * 100)
        print(f'*** Failed to process {failed_count:,d} file(s).', end=(' Error messages:' if len(failed_msgs) else ''))
        print('\n'.join(['***\t--->' + m for m in failed_msgs]))
        print('*' * 100)
    print('\nExecution complete.')
