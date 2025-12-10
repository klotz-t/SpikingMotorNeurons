import numpy as np
from scipy.signal import butter, filtfilt, fftconvolve, detrend 
from scipy.signal.windows import hann
from factor_analyzer import FactorAnalyzer, Rotator
from ..utils.core import surogate_spike_train, get_binary_spike_trains


def filtered_spike_trains_hann(spike_trains, fsamp, win_dur=0.4, high_pass=0.5, high_pass_order=2):
    """
    Filter binary spike trains using a hanning window  

    Args:
        spike_trains (ndarray): Binary spike trains 
        fsamp (float): Sampling frequency
        low_pass (float): Cut-off frequency for the low-pass filter 
        high_pass (float): Cut-off frequency for the high-pass filter

    Returns:
        ndarray : Filtered spike trains
    """

    win = hann(int(win_dur * fsamp)) 
    win = win / np.sum(win)

    filtered_spike_trains = np.zeros_like(spike_trains)

    for i in range(spike_trains.shape[0]):
        #filtered_spike_trains[i,:] = filtfilt(win, 1, spike_trains[i,:], axis=1)
        filtered_spike_trains[i,:] = fftconvolve(spike_trains[i,:], win, mode='same')

    if not high_pass is None:
        b, a = butter(high_pass_order, high_pass, fs=fsamp, btype='high')
        filtered_spike_trains = filtfilt(b, a, filtered_spike_trains, axis=1)

    return filtered_spike_trains

def estimate_num_factors_parallel_analysis(spike_times, 
                                           fsamp,
                                           n_samples,
                                           num_realizations=100, 
                                           prctile=95,
                                           win_dur=0.4,
                                           high_pass=0.5,
                                           high_pass_order=2,
                                           random_seed=12
                                           ):
    """
    Estimate the number of latent factors explaining filtered motor neuron 
    spike trains through parallel analysis

    Args:
        - spike_times (dict): Time stamps of the motor neuron firings
        - fsamp (float): Sampling frequency in Hz
        - n_samples (int): Number of samples of the binary spike trains
        - num_realizations (int): Number of random spike train realizations considered
        - prctile (float): Significance threshold 
        - window (str): TODO
        - win_length (float): Duration of the window function used for low-pass filtering in seconds
        - high_pass (float): Cut-off frequency for a high-pass filter to detrend filtered spike trains
        - high_pass_oder (int): Order of the applied high-pass filter
        - random_seed (int): Seed of the random number generator

    Returns:
        - eigenvalues(ndarray): Eigenvalues of the correlation or covariance matrix
        - rand_percentile (float): Threshold indicating which eigenvalues reach the requested significance level
    """
    
    # Init random number generator
    rng = np.random.seed(random_seed)
    rand_seeds = np.random.randint(10000, size=num_realizations)

    rand_eigenvalues = np.zeros((num_realizations, len(spike_times)))

    

    for i in range(num_realizations):

        rand_spike_times = surogate_spike_train(spike_times, random_seed=rand_seeds[i])
        rand_spike_trains = get_binary_spike_trains(rand_spike_times, n_samples)
        filtered_rand_spikes = filtered_spike_trains_hann(rand_spike_trains, 
                                                         fsamp, 
                                                         win_dur=win_dur,
                                                         high_pass=high_pass,
                                                         high_pass_order=high_pass_order)
        
        corr = np.corrcoef(filtered_rand_spikes.T, rowvar=False)
        rand_eigenvalues[i,:] = np.linalg.eigvals(corr)

    rand_percentile = np.percentile(rand_eigenvalues.flatten(), prctile)

    spike_trains = get_binary_spike_trains(spike_times, n_samples)
    filtered_spikes = filtered_spike_trains_hann(spike_trains, 
                                                fsamp, 
                                                win_dur=win_dur,
                                                high_pass=high_pass,
                                                high_pass_order=high_pass_order)
    
    corr = np.corrcoef(filtered_spikes.T, rowvar=False)

    eigenvalues = np.sort(np.linalg.eigvals(corr))[::-1]

    return eigenvalues, rand_percentile

                                            
def run_factor_analysis(filtered_spikes_trains, n_factors, method='ml', maxiter=100000, rot='promax'):

    # 1. Run factor analysis
    fa = FactorAnalyzer(
        n_factors=n_factors,
        rotation=None,       
        method=method,
    )
    # Fit the model
    fa.fit(filtered_spikes_trains.T)
    # Extract the loadings
    L = fa.loadings_

    # 2. Apply rotation
    rotator = Rotator(method=rot, max_iter=maxiter)
    L_rot = rotator.fit_transform(L)

    scores = filtered_spikes_trains.T @ L_rot  
    unique_vars = fa.get_uniquenesses()  

    _, _, cum_factor_var = fa.get_factor_variance()

    return L_rot, scores, unique_vars, cum_factor_var


