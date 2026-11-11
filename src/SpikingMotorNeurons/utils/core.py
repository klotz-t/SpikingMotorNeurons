import numpy as np
import pandas as pd

def get_binary_spike_trains(
        spikes: pd.DataFrame, 
        t_dur: float, 
        fsamp: float
):
    """
    Convert a table of  motor neuron spikes 
    into binary spike trains

    Parameters
    ----------

    spike times : pd.DataFrame 
        Table of motor neuron discharges with columns 
        ``onset``, ``unit_id`` and ``event_type``
    t_dur : float 
        Length of the signal in seconds
    fsamp : float
        Sampling rate in Hz

    Returns
    -------
    spike_trains : np.ndarray 
        Array of binary spike trains with shape
        ``(n_units, n_samples)``
    """
    units = sorted(spikes["unit_id"].unique())
    n_mn = len(units)

    n_samples = int(np.ceil((t_dur) * fsamp)) + 1
    spike_trains = np.zeros([n_mn, n_samples])

    for i, l1 in enumerate(units):
        mu_spikes = spikes[spikes["unit_id"] == l1]["onset"].values
        mu_spikes = mu_spikes[
            (mu_spikes >= 0) & (mu_spikes < t_dur)
        ]
        mu_spike_idx = np.round((mu_spikes) * fsamp).astype(int)
        spike_trains[i, mu_spike_idx] = 1

    return spike_trains

def surogate_spike_train(
        spikes: pd.DataFrame, 
        fsamp: float,
        random_seed: int = 13
):
    """
    Generate a random spike train by randomly shuffeling 
    the observed interspike intervals

    Parameters
    ----------
    spikes : pd.DataFrame 
        Table of motor neuron discharges with columns 
        ``onset``, ``unit_id`` and ``event_type``
    random_seed : int
        Seed for the random number generator

    Returns
    -------
    surogate_spikes : pd.DataFrame 
        Table of shuffled motor neuron discharges with columns 
        ``onset``, ``duration``, ``unit_id`` 
        and ``event_type``
        
    """
    
    rng = np.random.default_rng(random_seed)
    surogate_spikes = pd.DataFrame(
        columns=["onset", "duration", "unit_id", "event_type"]
    )

    units = sorted(spikes["unit_id"].unique())

    for unit in units:

        # random permutations of the ISI
        mu_spikes = spikes[spikes["unit_id"] == unit]["onset"].values
        isi = np.diff(mu_spikes)
        shuffled_isi = rng.permutation(isi)

        # Get the fisrt spike        
        t = mu_spikes[0] + rng.uniform(low=-isi.std(), high=isi.std())
        surogate_spikes.loc[len(spikes)] = [
            t, 0, unit, "motor-unit-spike"
        ]

        # Add randomly shuffled spikes
        for j in range(len(isi)):
            t = t + shuffled_isi[j]
            surogate_spikes.loc[len(spikes)] = [
                t, 0, unit, "motor-unit-spike"
            ]

        surogate_spikes.sort_values(
            by=["onset", "unit_id"],
            ignore_index=True
        )

    return surogate_spikes  