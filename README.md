# predictingterror
Public code for the paper "Predicting Terrorist Attacks in the United States using Localized News Data" by SJ Krieg et al.

To run the models described in the paper, you will need to do the following.
1. Extract features from GDELT. Run the following:
  ```
  python get_gdelt_features.py STARTDATE ENDDATE NUMWORKERS
  ```
  where STARTDATE and ENDDATE are strings representing the dates to be extracted in YYYYMMDD format
  and NUMWORKERS is an integer representing the number of CPU cores to use.
  WARNING: these are large downloads and will use a lot of network bandwidth. The script cleans up temp files as it goes, so disk space should not be an issue.
  Key dependencies for this script include pandas (tested with 1.3.5).
  The ref/ directory (and the files in it) must also be accessible from the same directory as the script.
2. Extract the relevant events from the [Global Terrorism Database](https://www.start.umd.edu/gtd/). At minimum this must be a CSV file with the columns "country_txt", "provstate",  "iyear", "imonth", and "iday". See the label_features() function inside utils.py for details.
3. Run one of the following scripts to train the appropriate model:
  ```
  python train_rf.py
  python train_ff1h.py
  python train_lstm.py
  ```
  Please see the source code (including comments) for information about package dependencies.
  You will likely need to use the arguments -if1 INFGDELT and -if2 INFGTD to specify the location of the GDELT features and GTD labels, or modify the default argument within the source code.
  The neural networks and ensembles can be easily modified within the source code using the Keras and sklearn API.
  Other command line arguments include:
  ```
  -d --maxdelta: int, default==14
  -c --cpus: int, default=mp.cpu_count()
  -l --locs: list of str, default=['NY','CA','TX','FL','WA']
  ```
  More arguments can be found within the source code.
  
  Please e-mail skrieg@nd.edu with any questions. :-)
