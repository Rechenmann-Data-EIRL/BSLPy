import functools
import math
import operator
import os

import pandas
import pynwb
from hdmf.common import DynamicTable
from pynwb.core import NWBTable, NWBDataInterface
from scipy import signal
import time
from pynwb import NWBHDF5IO, ProcessingModule
import pandas as pd
import numpy as np
import plotly.io as pio

from src.bsl_python.GUI.dashboard import Dashboard

pio.templates.default = "plotly_dark"
pd.options.plotting.backend = "plotly"


def flatten_dict(d):
    def items():
        for key, value in d.items():
            if isinstance(value, dict):
                for subkey, subvalue in flatten_dict(value).items():
                    yield key, subvalue
            else:
                yield key, value

    return dict(items())


def get_activity(activity, tmin, tmax):
    index_tmin = math.floor((tmin + 0.2) / 0.001)
    index_tmax = math.floor((tmax + 0.2) / 0.001)
    new_activity = {electrode: [activity[electrode][trial][index_tmin:index_tmax] for trial in
                                range(len(activity[electrode]))] for electrode in activity.keys()}
    return new_activity


def get_activity_for_electrode(activity, electrode, tmin, tmax, list_trials):
    if tmin is None or tmax is None:
        return [[0]] * list_trials
    index_tmin = math.floor((tmin + 0.2) / 0.001)
    index_tmax = math.floor((tmax + 0.2) / 0.001)
    new_activity = [activity[electrode][trial][index_tmin:index_tmax] for trial in
                    range(len(activity[electrode]))]
    return new_activity


def signal_width(waveform, waveform_reverse, position, threshold):
    pos_rev = len(waveform_reverse) - position
    half_dur = next((index for index in range(len(waveform[position:-1])) if waveform[position + index] > threshold),
                    len(waveform) - position)
    half_dur2 = next(
        (index for index in range(len(waveform_reverse[pos_rev:-1])) if waveform_reverse[pos_rev + index] > threshold),
        len(waveform) - pos_rev)
    return half_dur + half_dur2


def compute_waveform_analysis(units):
    waveform_analysis = {}
    nb_units = len(units)
    fs = 24414.0625 / 1000
    for unit_index in range(nb_units):
        waveform = units[unit_index].waveform_mean.tolist()[0].flatten()
        waveform_reverse = waveform[::-1]
        min_peak_value = min(waveform)
        min_peak_pos = np.where(waveform == min(waveform))[0][0]

        # baseline is taken on 20 values of the waveform before the peak
        baseline = np.mean(waveform[min_peak_pos - 31:min_peak_pos - 11])
        min_peak_amplitude = min_peak_value - baseline
        second_peak_amp = None
        peak_to_through = None
        peak_ratio = None
        slope = None
        width_half_peak = None
        peak_duration = None
        through_duration = None
        spike_dur = None
        if min_peak_pos < len(waveform) - 1:
            second_max_peak_val = max(waveform[min_peak_pos + 1:-1])
            second_max_peak_pos = np.where(waveform == second_max_peak_val)[0][0]
            second_peak_amp = second_max_peak_val - baseline
            peak_to_through = (second_max_peak_pos - min_peak_pos) * 1 / fs
            peak_ratio = min_peak_amplitude / second_peak_amp
            mid_slope_position = (min_peak_pos + second_max_peak_pos) / 2
            step = 0.5 if (min_peak_pos + second_max_peak_pos) % 2 else 1
            slope = (waveform[int(mid_slope_position + step)] - waveform[int(mid_slope_position - step)]) / (2 * step)
            # DEAL WITH THE WIDTH OF HALF PEAK:
            width_half_peak = signal_width(waveform, waveform_reverse, min_peak_pos, min_peak_value / 2) * 1 / fs
            peak_duration = signal_width(waveform, waveform_reverse, min_peak_pos, 0) * 1 / fs
            through_duration = signal_width(-waveform, -waveform_reverse, second_max_peak_pos, 0) * 1 / fs
            spike_dur = peak_duration + through_duration

        waveform_analysis[units.id[unit_index]] = {"pb_amp": min_peak_amplitude, "tb_amp": second_peak_amp,
                                                   "p2t": peak_to_through, "p_rat": peak_ratio, "slope": slope,
                                                   "width_half_peak": width_half_peak,
                                                   "peak_dur": peak_duration, "through_dur":
                                                       through_duration, "spike_dur": spike_dur}
    return waveform_analysis


def compute_tuning_curve_analysis(all_spikes):
    return {}


def create_module_from_activity(name, description, activity, frequency):
    module = ProcessingModule(name=name, description=description)
    for electrode in activity.keys():
        module.add(
            pynwb.base.TimeSeries('electrode' + str(electrode), activity[electrode], unit="spikes/s", rate=frequency,
                                  comments="Each row corresponds to a specific trial"))
    return module


def save_file(path_to_file, nwbfile):
    io = NWBHDF5IO(path_to_file, mode='w')
    io.write(nwbfile)
    io.close()


if __name__ == '__main__':
    start_process_time = time.process_time()
    path_to_file = os.path.join("C:/Users/jujud/Documents/Consulting/Data/191128EM/NWB", "191128EM_Block-1.nwb")
    nwb_io = NWBHDF5IO(path_to_file, 'r')
    nwbfile = nwb_io.read()
    nwb_io.close()
    spike_times = []
    unit_indices = []
    electrodes = []
    units = range(len(nwbfile.units))
    for unit_index in units:
        unit = nwbfile.units[unit_index]
        if len(unit) > 0:
            spikes = unit["spike_times"].tolist()[0].tolist()
            electrode = unit["electrodes"].tolist()[0].index[0]
            spike_times += spikes
            unit_indices += [nwbfile.units.id[unit_index]] * len(spikes)
            electrodes += [electrode] * len(spikes)
    trials = range(len(nwbfile.trials))
    trial_indices = [-1] * len(spike_times)
    spike_times_per_trial = [-1] * len(spike_times)
    additional_columns = [column for column in nwbfile.trials.colnames if column not in ["start_time", "stop_time"]]
    additional_data = dict()

    for key in additional_columns:
        additional_data[key] = [None] * len(spike_times)
    info = [flatten_dict(nwbfile.trials[trial_index].to_dict()) for trial_index in trials]
    for trial_index in trials:
        start_time = info[trial_index]["start_time"]
        spikes = [spike_index for spike_index in range(len(spike_times)) if
                  start_time - 0.2 < spike_times[spike_index] < start_time + 3.5]
        for spike in spikes:
            trial_indices[spike] = trial_index
            spike_times_per_trial[spike] = spike_times[spike] - start_time
            for column in additional_data.keys():
                additional_data[column][spike] = info[trial_index][column]
    d = {'spike_times': spike_times, 'electrodes': electrodes, "unit": unit_indices, 'trials': trial_indices,
         'trial_time': spike_times_per_trial, **additional_data}
    all_spikes = pd.DataFrame.from_dict(d)
    all_spikes.dropna(inplace=True)

    time_elapsed = (time.process_time() - start_process_time)
    print("Build dataframe - elapsed time ", time_elapsed)
    start_process_time = time.process_time()
    tmax = 1
    tmin = -0.2
    nBins = round((tmax - tmin) / 0.001)

    activity = dict()
    activity_0db = dict()
    stimulus_name = "decB"
    list_electrodes = range(len(nwbfile.electrodes))
    list_electrodes = [62]
    list_trials = list(np.unique(all_spikes["trials"]))
    for electrode in list_electrodes:
        electrode_spikes = all_spikes[all_spikes["electrodes"] == electrode]
        activity[electrode] = [np.histogram(electrode_spikes['trial_time'][(electrode_spikes["trials"] == trial)],
                                            nBins, range=(tmin, tmax))[0] * 1000 for trial in list_trials]
    time_elapsed = (time.process_time() - start_process_time)
    print("Compute activity rate - elapsed time ", time_elapsed)
    start_process_time = time.process_time()
    spontaneous_activity = get_activity(activity, -0.15, -0.05)
    post_stim_activity = get_activity(activity, 0.3, 0.5)

    mean_spontaneous_activity = dict()
    std_spontaneous_activity = dict()
    mean_post_stim_activity = dict()
    std_post_stim_activity = dict()
    for electrode in activity.keys():
        mean_spontaneous_activity[electrode] = np.mean(spontaneous_activity[electrode])
        std_spontaneous_activity[electrode] = np.std(spontaneous_activity[electrode])
        mean_post_stim_activity[electrode] = np.mean(post_stim_activity[electrode])
        std_post_stim_activity[electrode] = np.std(post_stim_activity[electrode])

    # A beta value of 14 is probably a good starting point. Note that as beta gets large, the window narrows,
    # and so the number of samples needs to be large enough to sample the increasingly narrow spike, otherwise NaNs
    # will get returned. Most references to the Kaiser window come from the signal processing literature, where it is
    # used as one of many windowing functions for smoothing values. It is also known as an apodization (which means
    # “removing the foot”, i.e. smoothing discontinuities at the beginning and end of the sampled signal) or tapering
    # function.
    time_elapsed = (time.process_time() - start_process_time)
    print("Compute specific activity rate elapsed time ", time_elapsed)
    start_process_time = time.process_time()
    window = np.hanning(9)  # window length of 15 points and a beta of 14
    window = window / np.sum(window)
    filtered_activity = {electrode: [signal.filtfilt(window, 1, activity[electrode][trial], method="gust").tolist() for
                                     trial in list_trials] for electrode in list_electrodes}

    mean_activity = {electrode: np.mean(np.array([activity[electrode][trial] for trial in list_trials]), axis=0) for
                     electrode in list_electrodes}
    mean_filtered_activity = {
        electrode: np.mean(np.array([filtered_activity[electrode][trial] for trial in list_trials]), axis=0) for
        electrode in list_electrodes}
    peak_amplitude = {electrode: np.max(mean_filtered_activity[electrode]) for electrode in list_electrodes}
    peak_latency = {
        electrode: (np.where(mean_filtered_activity[electrode] == peak_amplitude[electrode])[0][0] - 200) * 0.001
        for electrode in list_electrodes}
    waveform_analysis = compute_waveform_analysis(nwbfile.units)
    tuning_curve_analysis = compute_tuning_curve_analysis(all_spikes, )
    nStdDev = 1 / 4
    onset_min_time = 200
    onset_max_time = 450
    time_process = {electrode: [index for index, value in enumerate(
        mean_filtered_activity[electrode][onset_min_time:onset_max_time] >= (
                mean_spontaneous_activity[electrode] + nStdDev * std_spontaneous_activity[electrode])) if value] for
                    electrode in list_electrodes}
    onset = {electrode: time_process[electrode][0] * 0.001 if len(time_process[electrode]) > 0 else None for electrode
             in list_electrodes}
    offset = {electrode: time_process[electrode][-1] * 0.001 if len(time_process[electrode]) > 0 else None for electrode
              in list_electrodes}
    duration = {electrode: offset[electrode] - onset[electrode] if offset[electrode] is not None and onset[
        electrode] is not None else 1 for electrode in list_electrodes}
    stim_activity = {'activity': functools.reduce(operator.iconcat, [
        np.sum(get_activity_for_electrode(activity, electrode, 0.01, 0.06,
                                          np.max(list_trials) + 1),
               1) * 0.001 / 0.05 for electrode in list_electrodes], []),
                     'freq': functools.reduce(operator.iconcat,
                                              [[row["freq"] for row in info] for electrode in list_electrodes], []),
                     'decB': functools.reduce(operator.iconcat,
                                              [[row["decB"] for row in info] for electrode in list_electrodes], []),
                     'electrode': functools.reduce(operator.iconcat,
                                                   [[electrode] * len(info) for electrode in list_electrodes], [])}
    df = pd.DataFrame(stim_activity)
    trf = df.groupby(['electrode', 'freq', 'decB']).mean()

    time_elapsed = (time.process_time() - start_process_time)
    print("Compute parameters - elapsed time ", time_elapsed)
    all_spiking_activity_module = ProcessingModule(name='all_spiking_activities', description='Spiking activity')
    fs = 24414.0625 / 1000
    all_spiking_activity_module.add_container(
        create_module_from_activity(name='spiking_activity', description='Spiking activity per trial and electrode',
                                    activity=activity, frequency=fs))
    all_spiking_activity_module.add_container(create_module_from_activity(name='filtered_activity',
                                                                          description='Filtered spiking activity with hanning window per trial and electrode',
                                                                          activity=filtered_activity, frequency=fs))
    all_spiking_activity_module.add_container(create_module_from_activity(name='mean_spiking_activity',
                                                                          description='Average spiking activity per electrode',
                                                                          activity=mean_activity, frequency=fs))
    all_spiking_activity_module.add_container(create_module_from_activity(name='mean_filtered_activity',
                                                                          description='Average filtered spiking activity per electrode',
                                                                          activity=mean_filtered_activity,
                                                                          frequency=fs))
    activity_parameters = pd.DataFrame.from_dict({"electrode": list_electrodes,
                                                  "mean_spontaneous_activity": list(mean_spontaneous_activity.values()),
                                                  "std_spontaneous_activity": list(std_spontaneous_activity.values()),
                                                  "mean_post_stim": list(mean_post_stim_activity.values()),
                                                  "std_post_stim": list(std_post_stim_activity.values())})
    all_spiking_activity_module.add(DynamicTable.from_dataframe(activity_parameters, name="parameters"))

    nwbfile.add_processing_module(all_spiking_activity_module)
    save_file(path_to_file, nwbfile)
    Dashboard.create(all_spikes, list_electrodes, mean_spontaneous_activity, peak_amplitude, nwbfile.units, trf,
                     mean_filtered_activity)
