import numpy as np
from scipy.signal import butter, filtfilt, coherence, detrend
from scipy.optimize import least_squares


def get_mean_spike_coherence(st1, st2, fsamp=2048, f_low=0, f_up=5, win_width=1, zero_pad=10):
    """
    Get the mean coherecne of two (cumulative) spike trains using Welch's method 
    in a given frequency interval 

    Args:
        st1 (ndarray): Binary spike train 1
        st2 (ndarray): Binary spike train 2
        fsamp (float): Sampling frequency in Herz
        f_low (float): Lower bound of the frequecny interval of interest
        f_up (float): Upper bound of the frequecny interval of interest
        win_width (float): Width of the temporal window in seconds for conducting a FFT
        zero_pad (int): Zero-padding of FFT for increasing the frequency resolution

    Returns:
        mean_coherence (float) : Mean coherecne in a given frequency band
    """


    f, Cxy = coherence(detrend(st1), detrend(st2), fs=fsamp, noverlap=None,
                       nperseg=win_width*fsamp,nfft=fsamp*zero_pad, window='hann')
    
    idx1 = int(zero_pad*f_low)
    idx2 = int(zero_pad*f_up)

    mean_coherence = np.mean(Cxy[idx1:idx2])

    return mean_coherence

def get_z_scored_coherence(spike_trains, M=None, num_realizations=100, 
                           fsamp=2048, win_width=1, zero_pad=10, random_seed=42):
    
    rng = np.random.seed(random_seed)

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
        all_indices = np.random.choice(num_of_mus, size=2*M, replace=False)
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


def estimate_common_drive_index(spike_trains, num_realizations=30, fsamp=2048,
                                f_low=0, f_up=5, win_width=1, zero_pad=10, random_seed=42):
    
    """
    Estimate the common drive index (CDI) as proposed in Negro, Yavuz & Farina (2016)
    by computing the coherence between sub-populations of cummulative spike trains. 

    Args:
        spike_trains (ndarray): Matrix of binary spike train 1
        num_realizations (int): Number of random repetitions
        fsamp (float): Sampling frequency in Herz
        f_low (float): Lower bound of the frequecny interval of interest
        f_up (float): Upper bound of the frequecny interval of interest
        win_width (float): Width of the temporal window in seconds for conducting a FFT
        zero_pad (int): Zero-padding of FFT for increasing the frequency resolution
        random_seed (int): Initalize the random seed generator

    Returns:
        PCI (float) : Common drive index 
    """

    rng = np.random.seed(random_seed)

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
            all_indices = np.random.choice(num_of_mus, size=2*M, replace=False)
            subset1 = all_indices[:M]
            subset2 = all_indices[M:]

            # Get cumulative spike trains
            st1 = np.sum(spike_trains[subset1,:], axis=0)
            st2 = np.sum(spike_trains[subset2,:], axis=0)

            # Compute the coherence between the cumulative spike trains
            coherence_values[i,j] = get_mean_spike_coherence(st1,st2,fsamp=fsamp,f_low=f_low,f_up=f_up,
                                                             win_width=win_width,zero_pad=zero_pad)

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
    least_square_optimizer = least_squares(loss_function, [1, 1], loss='soft_l1', f_scale=0.1, bounds=(1e-3,np.inf),
                                           args=(pooled_mus.flatten(), coherence_values.flatten()))
    
    # Solve the optimization problem
    A_fit, B_fit = least_square_optimizer.x

    # Estimate the commond drive
    PCI = np.sqrt(A_fit / (A_fit + B_fit))

    return PCI