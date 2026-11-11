import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, fftconvolve, detrend 
from scipy.signal.windows import hann
from factor_analyzer import FactorAnalyzer, Rotator
from ..utils.core import surogate_spike_train, get_binary_spike_trains


def filtered_spike_trains_hann(
        spike_trains: np.ndarray, 
        fsamp: float, 
        win_dur: float = 0.4, 
        high_pass: float = 0.5, 
        high_pass_order: float = 2
):
    """
    Filter binary spike trains using a hanning window  

    Parameters
    ----------
    spike_trains : ndarray
       Array of binary spike trains with shape
       ``(n_units, n_samples)`` 
    fsamp : float 
        Sampling frequency in Hz
    high_pass : float or None , default 0.5 
        Cut-off frequency for the high-pass filter
        If None, no highpass filter is applied
    high_pass_order : int, default 2 
        Order for the butterwoth-type high-pass filter

    Returns
    -------
    filtered_spike_trains : ndarray
        Filtered spike trains

    """

    win = hann(int(win_dur * fsamp)) 
    win = win / np.sum(win)

    n_mn, n_samples = spike_trains.shape

    filtered_spike_trains = np.zeros((n_mn,n_samples))

    for i in range(spike_trains.shape[0]):
        #filtered_spike_trains[i,:] = filtfilt(win, 1, spike_trains[i,:], axis=1)
        filtered_spike_trains[i,:] = fftconvolve(spike_trains[i,:], win, mode='same')

    if not high_pass is None:
        b, a = butter(high_pass_order, high_pass, fs=fsamp, btype='high')
        filtered_spike_trains = filtfilt(b, a, filtered_spike_trains, axis=1)

    return filtered_spike_trains

def estimate_num_factors_parallel_analysis(
        spikes: pd.DataFrame, 
        fsamp: float,
        t_dur: float,
        num_realizations: int = 100, 
        prctile: float = 95,
        win_dur: float = 0.4,
        high_pass: float = 0.5,
        high_pass_order: int = 2,
        random_seed: int = 12
):
    """
    Estimate the number of latent factors explaining 
    filtered motor neuron spike trains through 
    parallel analysis

    Parameters
    ----------

    spike_times : pd.DataFrame 
        Time stamps of the motor neuron firings
    fsamp : float 
        Sampling frequency in Hz
    t_dur : float 
        Duration of the signal in seconds
    num_realizations : int , default 100 
        Number of random spike train realizations considered
    prctile : float , default 95 
        Significance threshold 
    win_dur : float , default 0.4 
        Duration of the hanning window used for 
        low-pass filtering in seconds
    high_pass : float , default 0.5 
        Cut-off frequency for a high-pass filter to 
        detrend filtered spike trains
    high_pass_oder : int , default 2 
        Order of the applied high-pass filter
    random_seed : int , default 12 
        Seed of the random number generator

    Returns
    -------
    eigenvalues : np.ndarray 
        Eigenvalues of the correlation or covariance matrix
    rand_percentile : float
        Threshold indicating which eigenvalues reach the 
        requested significance level

    """
    
    # Init random number generator
    rng = np.random.default_rng(random_seed)
    rand_seeds = rng.integers(10000, size=num_realizations)
    #rand_seeds = np.random.randint(10000, size=num_realizations)

    units = sorted(spikes["unit_id"].unique())
    n_units = len(units)

    rand_eigenvalues = np.zeros((num_realizations, n_units))

    for i in range(num_realizations):

        rand_spikes = surogate_spike_train(
            spikes, 
            random_seed=rand_seeds[i]
        )
        rand_spike_trains = get_binary_spike_trains(
            rand_spikes, t_dur, fsamp
        )
        filtered_rand_spikes = filtered_spike_trains_hann(
            rand_spike_trains, 
            fsamp, 
            win_dur=win_dur,
            high_pass=high_pass,
            high_pass_order=high_pass_order
        )
        
        corr = np.corrcoef(filtered_rand_spikes.T, rowvar=False)
        rand_eigenvalues[i,:] = np.linalg.eigvals(corr)

    rand_percentile = np.percentile(rand_eigenvalues.flatten(), prctile)

    spike_trains = get_binary_spike_trains(
        spikes, t_dur, fsamp
    )
    filtered_spikes = filtered_spike_trains_hann(
        spike_trains, 
        fsamp, 
        win_dur=win_dur,
        high_pass=high_pass,
        high_pass_order=high_pass_order
    )
    
    corr = np.corrcoef(filtered_spikes.T, rowvar=False)

    eigenvalues = np.sort(np.linalg.eigvals(corr))[::-1]

    return eigenvalues, rand_percentile

                                            
def run_factor_analysis(
        filtered_spikes_trains, 
        n_factors, 
        method='ml', 
        maxiter=100000, 
        rot='promax'
):

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


