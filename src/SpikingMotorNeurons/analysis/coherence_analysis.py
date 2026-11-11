import numpy as np
from scipy.signal import butter, filtfilt, coherence, detrend
from scipy.optimize import least_squares
from typing import List, Literal, Optional


def get_mean_spike_coherence(
        st1: np.ndarray, 
        st2: np.ndarray, 
        fsamp: float = 2048, 
        f_low: float = 0, 
        f_up: float = 5, 
        win_width: float = 1, 
        zero_pad: int = 10
    ):
    """
    Get the mean coherecne of two (cumulative) spike trains using Welch's method 
    in a given frequency interval 

    Parameters
    ----------
    st1 : ndarray 
        Binary spike train of neuron 1
    st2 : ndarray
        Binary spike train or neuron 2
    fsamp : float , default 2048
        Sampling frequency in Hz
    f_low : float , default 0 
        Lower bound of the frequecny interval of interest
    f_up : float, default 5 
        Upper bound of the frequecny interval of interest
    win_width : float , default 1 
        Width of the temporal window in seconds for conducting a FFT
    zero_pad : int , default 10 
        Zero-padding of FFT for increasing the frequency resolution

    Returns
    -------
    mean_coherence : float
        Mean coherecne in a given frequency band
    """


    f, Cxy = coherence(detrend(st1), detrend(st2), fs=fsamp, noverlap=None,
                       nperseg=win_width*fsamp,nfft=fsamp*zero_pad, window='hann')
    
    idx1 = int(zero_pad*f_low)
    idx2 = int(zero_pad*f_up)

    mean_coherence = np.mean(Cxy[idx1:idx2])

    return mean_coherence

def get_z_scored_coherence(
        spike_trains: np.ndarray, 
        M: int | None = None, 
        num_realizations: int = 100, 
        fsamp: float = 2048, 
        win_width: float = 1, 
        zero_pad: int = 10, 
        random_seed: int = 42
    ):
    """
    Calculate the z-scored coherence between pairs
    of cummulative spike trains

    Parameters
    ----------

    spike_trains : np.ndarray
        Array of binary spike trains with shape 
        ``(n_units, n_samples)``
    M : int or None , default None
        Number of units of used per cummulative spike train.
        If None, the floor of 50 % of the units is used
    num_realizations : int , default 100
        Number of realizations for the cummulative spike trains
    win_width : float , default 1 
            Width of the temporal window in seconds for conducting a FFT
    zero_pad : int , default 10 
        Zero-padding of FFT for increasing the frequency resolution
    random_seed : int , default 42
        Random seed for reproduciple random numbers

    Returns 
    -------

    f : np.ndarray
        Array of frequencies
    z_coh : np.ndarray
        Array of z-scored coherence values

    """
    
    rng = np.random.default_rng(random_seed)

    num_of_mus = spike_trains.shape[0]

    if M == None:
        M = int(np.floor(num_of_mus / 2))

    st1 = spike_trains[0,:]
    st2 = spike_trains[1,:] 

    idx1 = int(zero_pad*100)
    idx2 = int(zero_pad*250)

    L = np.floor(spike_trains.shape[1] / (win_width*fsamp))

    f, Cxy = coherence(detrend(st1), detrend(st2), fs=fsamp, noverlap=None,
                       nperseg=win_width*fsamp,nfft=fsamp*zero_pad, window='hann')

    z_coh = np. zeros((num_realizations, f.shape[0]))

    for j in range(num_realizations):
        # Randomly asign MUs to two subsets
        all_indices = rng.choice(num_of_mus, size=2*M, replace=False)
        subset1 = all_indices[:M]
        subset2 = all_indices[M:]

        # Get cumulative spike trains
        st1 = np.sum(spike_trains[subset1,:], axis=0)
        st2 = np.sum(spike_trains[subset2,:], axis=0)

        _, Cxy = coherence(
            detrend(st1), detrend(st2), fs=fsamp, noverlap=None,   
            nperseg=win_width*fsamp,nfft=fsamp*zero_pad, window='hann'
        )

        bias = np.mean(Cxy[idx1:idx2])
        

        z_coh[j,:] = np.sqrt(2*L) * np.arctanh(Cxy) - bias


    return f, z_coh


def estimate_common_drive_index(
        spike_trains, 
        num_realizations: int = 30, 
        fsamp: float = 2048,
        f_low: float = 0, 
        f_up: float = 5, 
        win_width: float = 1, 
        zero_pad: int = 10, 
        random_seed: int = 42
):
    
    """
    Estimate the common drive index (CDI) as proposed in Negro, Yavuz & Farina (2016)
    by computing the coherence between sub-populations of cummulative spike trains. 

    Parameters
    ----------

    spike_trains : np.ndarray
        Array of binary spike trains with shape 
        ``(n_units, n_samples)``
    num_realizations : int , default 30
        Number of realizations for the cummulative spike trains
    fsamp : float , default 30 
        Sampling frequency in Hz
    f_low : float, default 0 
        Lower bound of the frequecny interval of interest
    f_up : float , default 5 
        Upper bound of the frequecny interval of interest
    win_width : float , default 1
        Width of the temporal window in seconds for conducting a FFT
    zero_pad : int , default 10 
        Zero-padding of FFT for increasing the frequency resolution
    random_seed : int , default 42 
        Initalize the random seed generator

    Returns
    -------
    PCI : float
        Estimated common drive index 
    """

    rng = np.random.default_rng(random_seed)

    num_of_mus = spike_trains.shape[0]

    max_number_of_pooled_mns = int(np.floor(num_of_mus / 2))

    pooled_mus = np.zeros((max_number_of_pooled_mns, num_realizations))

    coherence_values = np.zeros((max_number_of_pooled_mns, num_realizations))

    for i in np.arange(max_number_of_pooled_mns):
        for j in np.arange(num_realizations):

            # Number of MUs per subset
            M = i+1
            pooled_mus[i,j] = M
        
            # Randomly asign MUs to two subsets
            all_indices = rng.choice(num_of_mus, size=2*M, replace=False)
            subset1 = all_indices[:M]
            subset2 = all_indices[M:]

            # Get cumulative spike trains
            st1 = np.sum(spike_trains[subset1,:], axis=0)
            st2 = np.sum(spike_trains[subset2,:], axis=0)

            # Compute the coherence between the cumulative spike trains
            coherence_values[i,j] = get_mean_spike_coherence(
                st1,
                st2,
                fsamp=fsamp,
                f_low=f_low,
                f_up=f_up,
                win_width=win_width,
                zero_pad=zero_pad
            )

    # Helper function describing the CDI model        
    def CDI_model(x,A,B):
        y_est = abs(x**2 * A)**2 / ((x*B) + ((x**2)*A))**2
        return y_est
    
    # Helper function for describing the loss
    def loss_function(params, x, y):
        A,B = params
        loss = y - CDI_model(x, A, B)
        return loss

    # Set up a least squares optimization problem
    least_square_optimizer = least_squares(
        fun=loss_function, 
        x0=[1, 1], 
        loss='soft_l1', 
        f_scale=0.1, 
        bounds=(1e-3,np.inf),
        args=(pooled_mus.flatten(), coherence_values.flatten())
    )
    
    # Solve the optimization problem
    A_fit, B_fit = least_square_optimizer.x

    # Estimate the commond drive
    PCI = np.sqrt(A_fit / (A_fit + B_fit))

    return PCI