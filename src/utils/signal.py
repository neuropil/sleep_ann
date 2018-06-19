import numpy as np
from tqdm import tqdm as tqdm
import mtspec.multitaper as mtm

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def bin_dataset(dataset,window_s=30,fs=1024,downsample_factor=1):
    total_seconds = len(dataset)//fs
    output = []
    total_slices = total_seconds//window_s
    for i in np.arange(total_slices):
        slc = slice(window_s*fs*i,window_s*fs*(i+1),downsample_factor)
        output.append(dataset[slc].tolist())
    return np.array(output)

def pspec(data,samp_freq=1,df=1):
    # data.shape -> (epoch, window_length)
    win = int(data.shape[1])
    tw = df*win/samp_freq/2
    
    # L is a rough estimate of number of tapers needed
    L = int(2*tw)-1

    tapers,lamb,theta = mtm.dpss(win,tw,L)
    tapers = np.swapaxes(tapers,0,1)
    fft_split_data = []
    freqs = np.fft.fftfreq(win,d=1/samp_freq)[1:win//2]
    
    for i in tqdm(np.arange(data.shape[0])):
        # Copy the window L, times
        # rep_data -> 
        rep_data = np.matlib.repmat(data[i],L,1)
        
        # Elementwise multiplication of taper
        split_data = np.multiply(rep_data,tapers)
        
        # fft of the tapered data
        fft_data = np.fft.fft(split_data)[:,1:win//2]*2
        
        # Take the abs and square the frequency spectra
        fft_split_data.append(np.abs(fft_data)**2)

    # fft_split_data.shape = [num_epochs, num_tapers, freq_power]
    # Average over the tapers to get [num_epochs,mean_freq_power]
    mtspec_data = np.array(fft_split_data).mean(axis=1)
    
    mtpspec = mtspec_data.swapaxes(0,1)
    
    return (mtpspec,freqs,tapers)

def periodogram(data,samp_freq=1):
    win = int(data.shape[1])
    freqs = np.fft.fftfreq(win,d=1/samp_freq)[1:win//2]
    fft_split_data = []
    for i in tqdm(np.arange(data.shape[0])):
        epoch_dat = data[i]*(1/np.sqrt(win))
        fft_data = np.fft.fft(epoch_dat)[1:win//2]*2
        fft_split_data.append(np.abs(fft_data)**2)
    
    spx_data = np.swapaxes(np.array(fft_split_data),0,1)

    return (spx_data,freqs)

def condense_spectrogram(spec_data,freq_bins):
    num_freq_bins = len(freq_bins)
    num_epochs = spec_data.shape[1]
    condensed_spxgm = np.empty((num_freq_bins,num_epochs))
    for i,idxs in enumerate(freq_bins):
        condensed_spxgm[i,:] = spec_data[idxs,:].sum(axis=0)
    
    return condensed_spxgm

def gen_freq_bins(freqs,freq_ranges):
    # Parameters
    # freq_ranges:  list
    #     List of tuples of frequency range (inclusive). e.x. (0,3)
    freq_index_bins = [(f_start<freqs) & (freqs<f_stop) for f_start, f_stop in freq_ranges]
    
    return freq_index_bins