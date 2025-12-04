import numpy as np

def get_binary_spike_trains(spike_times, n_samples):
    """
    Convert motor neuron spike indices into binary spike trains

    Args:
        spike times (dict): Time stamps of motor neuron discharges (samples)
        n_samples (float): Length of the signal (in samples)

    Returns:
        ndarray : Binary spike trains
    """

    n_mn = len(spike_times)

    spike_trains = np.zeros([n_mn, n_samples])

    for i in np.arange(n_mn):
        spike_trains[i,spike_times[i]] = 1

    return spike_trains

def surogate_spike_train(spike_times, random_seed=13):
    """
    Generate a random spike train by randomly shuffeling the observed interspike intervals

    Args:
        spike times (dict): Time stamps of motor neuron discharges (samples)
        time_param (float): Dictonary with temporal properties of the trial

    Returns:
        ndarray : Binary spike trains
    """
    
    rng = np.random.seed(random_seed)

    surogate_spikes = {}

    for i in range(len(spike_times)):

        # random permutations of the ISI
        isi = np.diff(spike_times[i])
        shuffled_isi = np.random.permutation(isi)

        sign = np.random.choice([1, -1])
        t = np.zeros(len(spike_times[i]))
        t[0] = spike_times[i][0] + sign * np.random.randint(10)

        for j in range(len(spike_times[i])-1):
            t[j+1] = t[j] + shuffled_isi[j]

        surogate_spikes[i] = t.astype(int)    

    return surogate_spikes  