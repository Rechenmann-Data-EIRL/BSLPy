import functools
import math
import operator
import os

import scipy
from scipy import signal
import time
from pynwb import NWBHDF5IO
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio

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


def raster_plot(all_spikes, electrode_list, fig):
    initial_variable = "decB"
    index = 0
    buttons = []

    for electrode in electrode_list:
        electrode_spikes = all_spikes[all_spikes["electrodes"] == electrode]
        x = electrode_spikes["trial_time"]
        repetition_per_cond = dict.fromkeys(np.unique(electrode_spikes[initial_variable]).tolist(), 0)
        repetition = [0] * len(electrode_spikes[initial_variable])
        for rep_index in range(len(repetition)):
            cond = electrode_spikes[initial_variable].tolist()[rep_index]
            repetition_per_cond[cond] += 1
            repetition[rep_index] = repetition_per_cond[cond]
        y = electrode_spikes[initial_variable].tolist()
        new_fig = go.Scatter(x=x, y=y, mode='markers')
        new_fig.visible = False
        if index == 0:
            new_fig.visible = True
        fig.add_trace(new_fig, row=2, col=2)
        electrode_visibility = [False] * len(electrode_list)
        electrode_visibility[index] = True
        buttons.append(dict(
            args=[{"visible": electrode_visibility}],
            label="Electrode " + str(electrode),
            method="restyle"
        ))
        index += 1
    fig.update_xaxes(range=[-0.2, 1])
    y_axis_selectors = []
    columns = sorted(list(all_spikes.columns))
    columns.remove("spike_times")
    columns.remove("electrodes")
    columns.remove("trial_time")
    for column in columns:
        y_axis_selectors.append(dict(
            args=[{"y": [list(all_spikes[all_spikes["electrodes"] == electrode][column]) for electrode in
                         electrode_list]},
                  {"title": 'Spike activity per ' + column,
                   "yaxis": {"title": {"text": column}}}],
            label=column,
            method="update"
        ))
    fig.update_yaxes(title_text=initial_variable, row=2, col=2)
    # Add dropdown
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                pad={"r": 10, "t": 0},
                showactive=True,
                x=0.01,
                xanchor="left",
                y=1.07,
                yanchor="top"
            ),
            dict(
                buttons=y_axis_selectors,
                direction="down",
                pad={"r": 10, "t": 0},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.07,
                yanchor="top"
            ),
        ]
    )


def psth_plot(all_spikes, electrode_list, fig):
    index = 0
    for electrode in electrode_list:
        electrode_spikes = all_spikes[all_spikes["electrodes"] == electrode]
        spikes = electrode_spikes.trial_time.tolist()
        new_fig = go.Histogram(x=spikes, xbins=dict(start=-0.2, end=3.5, size=(3.5 + 0.2) / 500))
        new_fig.visible = False
        if index == 0:
            new_fig.visible = True
        fig.add_trace(new_fig, row=3, col=2)
        fig.update_xaxes(title_text="Time (s)", row=3, col=2)
        fig.update_yaxes(title_text="Count", row=3, col=2)
        index += 1


def trf_plot(trf, spontaneous_rate_mean, peak_amplitude, fig):
    levels = trf.index.levels
    nb_cols = 4
    nb_rows = math.ceil(max(levels[0]) / nb_cols)
    index = 0
    new_fig = make_subplots(rows=nb_rows, cols=nb_cols, shared_xaxes=False,
                            shared_yaxes=False, x_title='Frequency (Hz)', y_title='Intensity (dB)', )
    trf_fig = make_subplots(rows=1, cols=3)
    for electrode in levels[0]:
        data = trf.iloc[
            list(range(index * len(levels[1]) * len(levels[2]), (index + 1) * len(levels[1]) * len(levels[2])))]
        trf_matrix = np.reshape(data.values, (len(levels[1]), len(levels[2]))).transpose()
        filtered_trf = scipy.signal.medfilt2d(trf_matrix, (3, 3))
        cleaned_trf = filtered_trf - spontaneous_rate_mean[electrode]
        cleaned_trf[cleaned_trf < peak_amplitude[electrode] * 0.2] = 0

        summed_cleaned_trf_freq = np.sum(cleaned_trf, 0).tolist()
        summed_cleaned_trf_intensity = np.sum(cleaned_trf, 1).tolist()
        index_BF = summed_cleaned_trf_freq.index(max(summed_cleaned_trf_freq))
        index_THRCF = next(
            (index for index in range(len(summed_cleaned_trf_intensity)) if summed_cleaned_trf_intensity[index] > 0),
            None)
        index_THR = next((index for index in range(len(cleaned_trf[:, index_BF])) if cleaned_trf[index, index_BF] > 0),
                         None)
        max_cleaned_trf = cleaned_trf[index_THRCF, :].max()
        index_CF = None
        if index_THRCF is not None:
            index_CF = next((index for index in range(len(cleaned_trf[index_THRCF, :])) if
                             cleaned_trf[index_THRCF, index] == max_cleaned_trf), None)
        all_channel_trf_map = go.Heatmap(z=trf_matrix, coloraxis="coloraxis", name="Electrode_" + str(electrode))
        trf_map = go.Heatmap(z=trf_matrix, coloraxis="coloraxis", name="Electrode_" + str(electrode))
        filtered_trf_map = go.Heatmap(z=filtered_trf, coloraxis="coloraxis", name="Electrode_" + str(electrode))
        cleaned_trf_map = go.Heatmap(z=cleaned_trf, coloraxis="coloraxis", name="Electrode_" + str(electrode))
        trf_map.visible = False
        filtered_trf_map.visible = False
        if index == 0:
            trf_map.visible = True
            filtered_trf_map.visible = True
            cleaned_trf_map.visible = True
        trf_fig.add_trace(trf_map, row=1, col=1)
        trf_fig.add_trace(filtered_trf_map, row=1, col=2)
        trf_fig.add_trace(cleaned_trf_map, row=1, col=3)
        # Add shapes
        if index_BF is not None and index_THR is not None:
            trf_fig.add_shape(
                # Line reference to the axes
                type="line", xref="x", yref="y", x0=index_BF, y0=index_THR, x1=index_BF, y1=len(levels[2]) - 1,
                line=dict(
                    color="Red",
                    width=3,
                ),
                row=1, col=3,
                visible=True if index == 0 else False
            )
        if index_CF is not None and index_THRCF is not None:
            trf_fig.add_shape(
                # Line reference to the axes
                type="circle", xref="x", yref="y", x0=index_CF - 0.5, y0=index_THRCF - 0.5, x1=index_CF + 0.5,
                y1=index_THRCF + 0.5,
                line=dict(
                    color="Red",
                    width=1,
                ),
                row=1, col=3, visible=True if index == 0 else False
            )
        row = electrode % nb_rows
        column = math.floor(electrode / nb_rows)
        new_fig.add_trace(all_channel_trf_map, row=row + 1, col=column + 1)
        index += 1
    new_fig.update_layout(coloraxis=dict(colorscale='Haline'), showlegend=False)
    new_fig.update_yaxes(showgrid=False, zeroline=False, linecolor='black', showticklabels=False, ticks='')
    new_fig.update_xaxes(showgrid=False, zeroline=False, linecolor='black', showticklabels=False, ticks='')
    new_fig.show()
    trf_fig.update_layout(coloraxis=dict(colorscale='Haline'), showlegend=False)
    trf_fig.update_yaxes(showgrid=False, zeroline=False, linecolor='black', showticklabels=False, ticks='')
    trf_fig.update_xaxes(showgrid=False, zeroline=False, linecolor='black', showticklabels=False, ticks='')
    trf_fig.show()


def waveform_plot(units, electrode_list, fig, unit_index=None):
    if unit_index is not None:
        unit = unit_index
    fs = 24414.0625 / 1000
    x = (np.array(range(101)) - 50) / fs
    x_rev = x[::-1]
    nb_units = len(units)
    index = 0
    for electrode in electrode_list:
        unit_indices = {units.id[index]: index for index in range(nb_units) if
                        len(nwbfile.units[index].electrodes.index.values) > 0 and
                        (nwbfile.units[index].electrodes.values[0]["imp"] == -electrode).any()}
        if len(unit_indices) > 0:
            if unit_index is None:
                unit = unit_indices[list(unit_indices.keys())[0]]
            waveform_mean = nwbfile.units[unit].waveform_mean.values.tolist()[0].flatten()
            waveform_std = nwbfile.units[unit].waveform_sd.values.tolist()[0].flatten()
            y_upper = np.add(waveform_mean, waveform_std)
            y_lower = np.subtract(waveform_mean, waveform_std)
            mean_fig = go.Scatter(x=x, y=waveform_mean, mode="lines")
            y1 = list(y_upper) + list(y_lower[::-1])
            std_fig = go.Scatter(
                x=list(x) + list(x_rev),
                y=y1,
                fill='toself',
                fillcolor='rgba(0,100,80,0.2)',
                line_color='rgba(255,255,255,0)',
                showlegend=False
            )
            mean_fig.visible = False
            std_fig.visible = False
            if index == 0:
                mean_fig.visible = True
                std_fig.visible = True
            index += 1
            fig.add_trace(mean_fig, row=2, col=3)
            fig.add_trace(std_fig, row=2, col=3)


def get_activity_rate(activity_rate, tmin, tmax):
    index_tmin = math.floor((tmin + 0.2) / 0.001)
    index_tmax = math.floor((tmax + 0.2) / 0.001)
    new_activity_rate = {electrode: [activity_rate[electrode][trial][index_tmin:index_tmax] for trial in
                                     range(len(activity_rate[electrode]))] for electrode in activity_rate.keys()}
    return new_activity_rate


def get_activity_rate_for_electrode(activity_rate, electrode, tmin, tmax, list_trials):
    if tmin is None or tmax is None:
        return [[0]] * list_trials
    index_tmin = math.floor((tmin + 0.2) / 0.001)
    index_tmax = math.floor((tmax + 0.2) / 0.001)
    new_activity_rate = [activity_rate[electrode][trial][index_tmin:index_tmax] for trial in
                         range(len(activity_rate[electrode]))]
    return new_activity_rate


def create_dashboard(all_spikes, electrode_list, spontaneous_rate_mean, peak_amplitude, units):
    start_process_time = time.process_time()
    fig = make_subplots(rows=3, cols=3, column_widths=[0.001, 0.699, 0.3], row_heights=[0.01, 0.59, 0.4],
                        vertical_spacing=0.02, print_grid=True)
    fig.update_layout(margin=dict(l=50, r=50, b=50, t=100))
    raster_plot(all_spikes, electrode_list, fig)
    psth_plot(all_spikes, electrode_list, fig)
    trf_plot(trf, spontaneous_rate_mean, peak_amplitude, fig)
    waveform_plot(units, electrode_list, fig)
    fig.update_layout(showlegend=False)
    index = 0
    for electrode in list_electrodes:
        new_fig = go.Scatter(x=np.linspace(-0.2, 1, int(1.2 / 0.001)), y=mean_filtered_activity_rate[electrode])
        new_fig.visible = False
        if index == 0:
            new_fig.visible = True
        fig.add_trace(new_fig, row=3, col=2)
        index += 1
    time_elapsed = (time.process_time() - start_process_time)
    print("Plot - elapsed time ", time_elapsed)
    fig.show()


def signal_width(waveform, waveform_reverse, position, threshold):
    pos_rev = len(waveform_reverse) - position
    half_dur = next((index for index in range(len(waveform[position:-1])) if waveform[position + index] > threshold), len(waveform) - position)
    half_dur2 = next((index for index in range(len(waveform_reverse[pos_rev:-1])) if waveform_reverse[pos_rev + index] > threshold), len(waveform) - pos_rev)
    return half_dur + half_dur2


def compute_waveform_analysis(nwbfile):
    waveform_analysis = {}
    nb_units = len(nwbfile.units)
    fs = 24414.0625 / 1000
    for unit_index in range(nb_units):
        waveform = nwbfile.units[unit_index].waveform_mean.tolist()[0].flatten()
        waveform_reverse = waveform[::-1]
        min_peak_value = min(waveform)
        min_peak_pos = np.where(waveform == min(waveform))[0][0]
        baseline = np.mean(waveform[
                           min_peak_pos - 31:min_peak_pos - 11])  # baseline is taken on 20 values of the waveform before the peak
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

        waveform_analysis[nwbfile.units.id[unit_index]] = {"pb_amp": min_peak_amplitude, "tb_amp": second_peak_amp,
                                                           "p2t": peak_to_through, "p_rat": peak_ratio, "slope": slope,
                                                           "width_half_peak": width_half_peak,
                                                           "peak_dur": peak_duration, "through_dur":
                                                               through_duration, "spike_dur": spike_dur}
    return waveform_analysis


def compute_tuning_curve_analysis(all_spikes):
    return {}


if __name__ == '__main__':
    start_process_time = time.process_time()
    nwb_io = NWBHDF5IO(os.path.join("C:/Users/jujud/Documents/Consulting/Data/191128EM/NWB", "191128EM_Block-1.nwb"),
                       'r')
    nwbfile = nwb_io.read()
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

    activity_rate = dict()
    activity_rate_0db = dict()
    stimulus_name = "decB"
    list_electrodes = range(len(nwbfile.electrodes))
    list_electrodes = [62]
    list_trials = list(np.unique(all_spikes["trials"]))
    for electrode in list_electrodes:
        electrode_spikes = all_spikes[all_spikes["electrodes"] == electrode]

        activity_rate[electrode] = [np.histogram(
            electrode_spikes['trial_time'][(electrode_spikes["trials"] == trial)], nBins, range=(tmin, tmax))[0] * 1000
                                    for
                                    trial
                                    in
                                    list_trials]
    time_elapsed = (time.process_time() - start_process_time)
    print("Compute activity rate - elapsed time ", time_elapsed)
    start_process_time = time.process_time()
    spontaneous_activity = get_activity_rate(activity_rate, -0.15, -0.05)
    post_stim_activity = get_activity_rate(activity_rate, 0.3, 0.5)

    mean_spontaneous_activity_rate = dict()
    std_spontaneous_activity_rate = dict()
    mean_post_stim_activity_rate = dict()
    std_post_stim_activity_rate = dict()
    for electrode in activity_rate.keys():
        mean_spontaneous_activity_rate[electrode] = np.mean(spontaneous_activity[electrode])
        std_spontaneous_activity_rate[electrode] = np.std(spontaneous_activity[electrode])
        mean_post_stim_activity_rate[electrode] = np.mean(post_stim_activity[electrode])
        std_post_stim_activity_rate[electrode] = np.std(post_stim_activity[electrode])
    # activity_rate_null_stim = {electrode:[activity_rate[electrode][trial] for trial in list_trials if all_spikes[all_spikes["electrodes"] == electrode & all_spikes[""]]] for electrode in activity_rate.keys()}

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
    filtered_activity_rate = {
        electrode: [signal.filtfilt(window, 1, activity_rate[electrode][trial], method="gust").tolist() for
                    trial in list_trials] for electrode in list_electrodes}

    mean_activity_rate = {
        electrode: np.mean(np.array([activity_rate[electrode][trial] for trial in list_trials]), axis=0) for
        electrode in list_electrodes}
    mean_filtered_activity_rate = {
        electrode: np.mean(np.array([filtered_activity_rate[electrode][trial] for trial in list_trials]), axis=0) for
        electrode in list_electrodes}
    peak_amplitude = {electrode: np.max(mean_filtered_activity_rate[electrode]) for electrode in list_electrodes}
    peak_latency = {
        electrode: (np.where(mean_filtered_activity_rate[electrode] == peak_amplitude[electrode])[0][0] - 200) * 0.001
        for
        electrode in list_electrodes}
    waveform_analysis = compute_waveform_analysis(nwbfile)
    tuning_curve_analysis = compute_tuning_curve_analysis(all_spikes, )
    print(waveform_analysis[0]["pb_amp"])
    nStdDev = 1 / 4
    onset_min_time = 200
    onset_max_time = 450
    time_process = {electrode: [index for index, value in enumerate(
        mean_filtered_activity_rate[electrode][onset_min_time:onset_max_time] >= (
                mean_spontaneous_activity_rate[electrode] + nStdDev * std_spontaneous_activity_rate[electrode])) if
                                value] for electrode in list_electrodes}
    onset = {electrode: time_process[electrode][0] * 0.001 if len(time_process[electrode]) > 0 else None for electrode
             in list_electrodes}
    offset = {electrode: time_process[electrode][-1] * 0.001 if len(time_process[electrode]) > 0 else None for electrode
              in list_electrodes}
    duration = {electrode: offset[electrode] - onset[electrode] if
    offset[electrode] is not None and onset[electrode] is not None else 1 for electrode in list_electrodes}
    stim_activity = {'activity': functools.reduce(operator.iconcat, [
        np.sum(get_activity_rate_for_electrode(activity_rate, electrode, 0.01, 0.06,
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
    create_dashboard(all_spikes, list_electrodes, mean_spontaneous_activity_rate, peak_amplitude, nwbfile.units)
