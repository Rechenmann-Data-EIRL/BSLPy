from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import scipy
import time
import math


class Dashboard:
    @staticmethod
    def create(all_spikes, electrode_list, spontaneous_rate_mean, peak_amplitude, units, trf, mean_filtered_activity):
        start_process_time = time.process_time()
        fig = make_subplots(rows=3, cols=3, column_widths=[0.001, 0.699, 0.3], row_heights=[0.01, 0.59, 0.4],
                            vertical_spacing=0.02, print_grid=True)
        fig.update_layout(margin=dict(l=50, r=50, b=50, t=100))
        raster_plot(all_spikes, electrode_list, fig)
        psth_plot(all_spikes, mean_filtered_activity, electrode_list, fig)
        trf_plot(trf, spontaneous_rate_mean, peak_amplitude)
        waveform_plot(units, electrode_list, fig)

        fig.update_layout(showlegend=False)

        time_elapsed = (time.process_time() - start_process_time)
        print("Plot - elapsed time ", time_elapsed)
        fig.show()


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


def psth_plot(all_spikes, mean_filtered_activity_rate, electrode_list, fig):
    index = 0
    for electrode in electrode_list:
        electrode_spikes = all_spikes[all_spikes["electrodes"] == electrode]
        spikes = electrode_spikes.trial_time.tolist()
        psth_fig = go.Histogram(x=spikes, xbins=dict(start=-0.2, end=3.5, size=(3.5 + 0.2) / 500))
        average_activity_fig = go.Scatter(x=np.linspace(-0.2, 1, int(1.2 / 0.001)), y=mean_filtered_activity_rate[electrode])
        psth_fig.visible = False
        average_activity_fig.visible = False
        if index == 0:
            psth_fig.visible = True
            average_activity_fig.visible = True
        fig.add_trace(psth_fig, row=3, col=2)
        fig.add_trace(average_activity_fig, row=3, col=2)
        fig.update_xaxes(title_text="Time (s)", row=3, col=2)
        fig.update_yaxes(title_text="Count", row=3, col=2)
        index += 1


def trf_plot(trf, spontaneous_rate_mean, peak_amplitude):
    levels = trf.index.levels
    nb_cols = 4
    nb_rows = math.ceil(max(levels[0]) / nb_cols)
    index = 0
    new_fig = make_subplots(rows=nb_rows, cols=nb_cols, shared_xaxes=False,
                            shared_yaxes=False, x_title='Frequency (Hz)', y_title='Intensity (dB)')
    trf_fig = make_subplots(rows=1, cols=3)
    for electrode in levels[0]:
        data = trf.iloc[
            list(range(index * len(levels[1]) * len(levels[2]), (index + 1) * len(levels[1]) * len(levels[2])))]
        trf_matrix = np.reshape(data.values, (len(levels[1]), len(levels[2]))).transpose()

        all_channel_trf_map = go.Heatmap(z=trf_matrix, coloraxis="coloraxis", name="Electrode_" + str(electrode))
        add_trf_fig_for_electrode(electrode, index, levels, peak_amplitude, spontaneous_rate_mean, trf_fig, trf_matrix)
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


def add_trf_fig_for_electrode(electrode, index, levels, peak_amplitude, spontaneous_rate_mean, trf_fig, trf_matrix):
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
    if index_BF is not None and index_THR is not None:
        trf_fig.add_shape(
            type="line", xref="x", yref="y", x0=index_BF, y0=index_THR, x1=index_BF, y1=len(levels[2]) - 1,
            line=dict(color="Red", width=3),
            row=1, col=3,
            visible=True if index == 0 else False
        )
    if index_CF is not None and index_THRCF is not None:
        trf_fig.add_shape(
            type="circle", xref="x", yref="y", x0=index_CF - 0.5, y0=index_THRCF - 0.5, x1=index_CF + 0.5,
            y1=index_THRCF + 0.5,
            line=dict(color="Red", width=1),
            row=1, col=3, visible=True if index == 0 else False
        )


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
                        len(units[index].electrodes.index.values) > 0 and
                        (units[index].electrodes.values[0]["imp"] == -electrode).any()}
        if len(unit_indices) > 0:
            if unit_index is None:
                unit = unit_indices[list(unit_indices.keys())[0]]
            waveform_mean = units[unit].waveform_mean.values.tolist()[0].flatten()
            waveform_std = units[unit].waveform_sd.values.tolist()[0].flatten()
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
