# Marcel
# Load data


import numpy as np
import pandas as pd
import os



def load_data(hist_range=(0,40), n_bins=100):
   
    # Load Schütt et al. data
    PATH = './data/Schütt/PotsdamCorpusFixations.dat'
    df_schütt = pd.read_csv(PATH, header=0, delimiter=' ')
    r_schütt = df_schütt['sacamplitude'].tolist()


    schütt_trials = []
    for n_trial in np.arange(1,121):
        r = df_schütt[df_schütt['trial']==n_trial]['sacamplitude']
        schütt_trials.append(r)
        
    # Read filenames of processed Bambach et al. data
    PATH = './data/Bambach-processed/'
    files = [f for f in os.listdir(PATH) if os.path.isfile(PATH + f)]

    files_child = []
    files_parent=[]
    for name in files:
        if name.split('_')[4] == 'child':
            files_child.append(name)
        else:
            files_parent.append(name)


    # stack trials of Bambach children, Bambach parents and Schütt 
    y_ch = []
    for f in files_child:
        df = pd.read_csv(PATH+f, header=0, delimiter=',')
        r = df['amplitude']
        hist = np.histogram(r, range=hist_range, bins=n_bins, density=True)
        y_ch.append(hist[0])
    stack_ch = np.stack(y_ch)

    y_pa = []
    for f in files_parent:
        df = pd.read_csv(PATH+f, header=0, delimiter=',')
        r = df['amplitude']
        hist = np.histogram(r, range=hist_range, bins=n_bins, density=True)
        y_pa.append(hist[0])
    stack_pa = np.stack(y_pa)


    y_schütt = []
    for r in schütt_trials:
        hist = np.histogram(r, range=hist_range, bins=n_bins, density=True)
        y_schütt.append(hist[0])
    stack_schütt = np.stack(y_schütt)


    # Compute mean and std
    mean_ch = []
    error_ch = []
    for i in range(stack_ch.shape[1]):
        mean_ch.append(stack_ch[:,i].mean())
        error_ch.append(stack_ch[:,i].std())  
    mean_ch = np.array(mean_ch)
    error_ch = np.array(error_ch)


    mean_pa = []
    error_pa = []
    for i in range(stack_pa.shape[1]):
        mean_pa.append(stack_pa[:,i].mean())
        error_pa.append(stack_pa[:,i].std())
    mean_pa = np.array(mean_pa)
    error_pa = np.array(error_pa)


    mean_schütt = []
    error_schütt = []
    for i in range(stack_pa.shape[1]):
        mean_schütt.append(stack_schütt[:,i].mean())
        error_schütt.append(stack_schütt[:,i].std())
    mean_schütt = np.array(mean_schütt)
    error_schütt = np.array(error_schütt)


    dx = hist[1][1] - hist[1][0]
    xdata = hist[1][:-1]+dx/2

    return xdata, (mean_ch, error_ch), (mean_pa, error_pa), (mean_schütt, error_schütt)