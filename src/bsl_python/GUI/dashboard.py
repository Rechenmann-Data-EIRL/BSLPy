import os
import pandas
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import scipy.signal
import time
import math
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from pynwb import NWBHDF5IO
from src.bsl_python.GUI.index import ExperimentInfo
import dash_table
from src.bsl_python.GUI.app import app
from src.bsl_python.lab_book_loader import LabBookLoader

nwbfile = None
nwbfiles = []
c_file = None


class Dashboard:
    @staticmethod
    def create(new_nwbfile, new_nwbfiles, notebook):
        global nwbfile, nwbfiles, app, c_file
        nwbfile = new_nwbfile
        nwbfiles = new_nwbfiles
        c_file = list(nwbfiles.keys())[0]
        activity_parameters = nwbfile.processing["parameters"].data_interfaces[
            "activity_parameters"].to_dataframe().set_index("electrode")
        electrode_list = list(activity_parameters.index.values)
        app.layout = html.Div(id="dashboard",
                              children=[Dashboard.create_layout(activity_parameters, electrode_list, electrode_list[0], notebook),
                                        dbc.Toast(
                                            [html.P("This is the content of the toast", className="mb-0")],
                                            id="warning-toast",
                                            header="Warning",
                                            icon="danger",
                                            dismissable=True,
                                            is_open=False,
                                            style={"position": "fixed", "top": 10, "right": 10, "width": 350})
                                        ])

    @staticmethod
    def create_layout(activity_parameters, electrode_list, electrode, notebook):
        global nwbfile
        trf = nwbfile.processing["trf"].data_interfaces["trf"].to_dataframe().groupby(
            ['electrode', 'freq', 'decB']).mean()
        all_channel_trf_fig = all_channels_trf(trf, electrode_list)
        params = pandas.DataFrame.from_dict({"Parameter": activity_parameters.columns,
                                             "Value": activity_parameters.loc[int(electrode)].values.round(2)})
        file_list = []
        block_names = sorted(list(nwbfiles.keys()))

        for index in range(len(nwbfiles)):
            file_list.append({'label': notebook['Trials']['StimulusSet'][index] + " - " + block_names[index], 'value': block_names[index]})
        return dbc.Row(children=[
            dbc.Col(children=[
                "Blocks",
                dcc.Dropdown(
                    id='file-dropdown',
                    options=file_list,
                    value=c_file,
                    style={"color": "black"},
                    className="mb-1"
                ),
                "Electrodes",
                dcc.Dropdown(
                    id='electrode-dropdown',
                    options=[{'label': str(electrode), 'value': electrode} for electrode in electrode_list],
                    value=electrode,
                    style={"color": "black"},
                    className="mb-2"
                ),
                html.Hr(style={"border": "1px dashed white"}),
                ExperimentInfo.get_html(nwbfile),
                html.Hr(style={"border": "1px dashed white"}),
                html.Div(
                    id='parameters',
                    children=[dash_table.DataTable(
                        id='parameters_table',
                        columns=[{"name": i, "id": i} for i in params.columns],
                        data=params.to_dict('records'),
                        style_cell={"color": "black",
                                    'overflow': 'hidden',
                                    'textOverflow': 'ellipsis',
                                    'maxWidth': 0}
                    )])
            ], width={"size": 2}),
            dbc.Col(id="middle_panel",
                    children=Dashboard.create_middle_panel_figures(electrode_list[0]),
                    width={"size": 6}),
            dbc.Col(children=[
                dcc.Graph(
                    id='all_channel_trf',
                    figure=all_channel_trf_fig,
                    style={"height": "98vh"}
                )],
                width={"size": 4})
        ],
            className="m-1")

    @staticmethod
    def create_middle_panel_figures(electrode):
        start_process_time = time.process_time()
        all_spikes = nwbfile.processing["spikes"].data_interfaces["spikes"].to_dataframe()
        trf = nwbfile.processing["trf"].data_interfaces["trf"].to_dataframe().groupby(
            ['electrode', 'freq', 'decB']).mean()
        mean_filtered_activity = nwbfile.processing["mean_filtered_activity"].data_interfaces
        raster_fig = raster_plot(all_spikes, electrode)
        psth_fig = psth_plot(all_spikes, mean_filtered_activity, electrode)
        cleaned_trf = nwbfile.processing["trf"].data_interfaces["processed_trf"]["cleaned_trf"]
        filtered_trf = nwbfile.processing["trf"].data_interfaces["processed_trf"]["filtered_trf"]
        trf_parameters = nwbfile.processing["trf"].data_interfaces["trf_parameters"].to_dataframe()
        trf_fig = trf_plot(trf, cleaned_trf, filtered_trf, trf_parameters, electrode)

        waveform_fig = waveform_plot(nwbfile.units, electrode)
        print("Final Plot - elapsed time ", (time.process_time() - start_process_time))
        return [
            dcc.Graph(
                id='raster',
                figure=raster_fig,
                style={"height": "38vh"}
            ),
            dcc.Graph(
                id='psth',
                figure=psth_fig,
                style={"height": "20vh"}
            ),
            dcc.Graph(
                id='trf',
                figure=trf_fig,
                style={"height": "20vh"}
            ),
            dcc.Graph(
                id='waveform',
                figure=waveform_fig,
                style={"height": "20vh"}
            )
        ]


@app.callback(
    [dash.dependencies.Output('middle_panel', 'children'),
     dash.dependencies.Output('all_channel_trf', 'figure'),
     dash.dependencies.Output('parameters', 'children'),
     dash.dependencies.Output('warning-toast', 'is_open'),
     dash.dependencies.Output('warning-toast', 'children')],
    [dash.dependencies.Input('electrode-dropdown', 'value'), dash.dependencies.Input('file-dropdown', 'value')],
    prevent_initial_call=True)
def change_electrode(electrode, file):
    global c_file, nwbfile
    all_channel_trf_fig = dash.no_update
    changed_file = False
    if file != c_file:
        c_file = file
        nwb_io = NWBHDF5IO(nwbfiles[c_file], 'r')
        nwbfile = nwb_io.read()
        changed_file = True
    if len(nwbfile.processing) > 0:
        activity_parameters = nwbfile.processing["parameters"].data_interfaces[
            "activity_parameters"].to_dataframe().set_index("electrode")
        if changed_file:
            trf = nwbfile.processing["trf"].data_interfaces["trf"].to_dataframe().groupby(
                ['electrode', 'freq', 'decB']).mean()
            electrode_list = list(activity_parameters.index.values)
            all_channel_trf_fig = all_channels_trf(trf, electrode_list)
        params = pandas.DataFrame.from_dict({"Parameter": activity_parameters.columns,
                                             "Value": activity_parameters.loc[int(electrode)].values.round(2)})
        return Dashboard.create_middle_panel_figures(electrode), all_channel_trf_fig, [dash_table.DataTable(
            id='parameters_table',
            columns=[{"name": i, "id": i} for i in params.columns],
            data=params.to_dict('records'),
            style_cell={"color": "black"}
        )], dash.no_update, dash.no_update
    return [], all_channel_trf_fig, [], True, [html.P("No pre-processing data was found for this block", className="mb-0")]


def raster_plot(all_spikes, electrode):
    initial_variable = "decB"
    raster_fig = go.Figure()
    electrode_spikes = all_spikes[all_spikes["electrodes"] == electrode]
    x = electrode_spikes["trial_time"]
    repetition_per_cond = dict.fromkeys(np.unique(electrode_spikes[initial_variable]).tolist(), 0)
    repetition = [0] * len(electrode_spikes[initial_variable])
    for rep_index in range(len(repetition)):
        cond = electrode_spikes[initial_variable].tolist()[rep_index]
        repetition_per_cond[cond] += 1
        repetition[rep_index] = repetition_per_cond[cond]
    y_step = np.min(np.diff(np.unique(electrode_spikes[initial_variable].tolist())))*0.9 if len(electrode_spikes) > 0 else 0
    y = electrode_spikes[initial_variable].tolist() + repetition/np.max(repetition) * y_step - y_step/2 if len(electrode_spikes) > 0 else []
    raster_fig.add_trace(go.Scatter(x=x, y=y, mode='markers', marker=dict(size=2)))
    y_axis_selectors = []
    columns = sorted(list(all_spikes.columns))
    columns.remove("spike_times")
    columns.remove("electrodes")
    columns.remove("trial_time")
    for column in columns:
        y_axis_selectors.append(dict(
            args=[{"y": list(all_spikes[all_spikes["electrodes"] == electrode][column])},
                  {"title": 'Spike activity per ' + column,
                   "yaxis": {"title": {"text": column}}}],
            label=column,
            method="update"
        ))
    raster_fig.update_layout(xaxis=dict(
        range=(-0.2, 1),
        constrain='domain'
    ), yaxis_title=initial_variable)
    raster_fig.update_layout(margin=dict(l=50, r=50, b=10, t=10))
    raster_fig.update_layout(showlegend=False, template="plotly_dark")
    return raster_fig


def psth_plot(all_spikes, mean_filtered_activity_rate, electrode):
    new_fig = go.Figure()
    electrode_spikes = all_spikes[all_spikes["electrodes"] == electrode]
    spikes = electrode_spikes.trial_time.tolist()
    psth_fig = go.Histogram(x=spikes, xbins=dict(start=-0.2, end=3.5, size=(3.5 + 0.2) / 500), name="activity rate")
    average_activity_fig = go.Scatter(x=np.linspace(-0.2, 1, int(1.2 / 0.001)),
                                      y=mean_filtered_activity_rate["electrode" + str(electrode)].data,
                                      name="average activity (spike/s)")
    new_fig.add_trace(psth_fig)
    new_fig.add_trace(average_activity_fig)
    new_fig.update_xaxes(title_text="Time (s)")
    new_fig.update_yaxes(title_text="Count")
    new_fig.update_layout(xaxis=dict(
        range=(-0.2, 1),
        constrain='domain'
    ), yaxis_showgrid=True, margin=dict(l=50, r=50, b=10, t=10), legend=dict(
        orientation="h",
        yanchor="bottom",
        y=0.92,
        xanchor="right",
        x=1
    ), template="plotly_dark")
    return new_fig


def trf_plot(trf, cleaned_trf, filtered_trf, trf_parameters, electrode):
    levels = trf.index.levels
    trf_fig = make_subplots(rows=1, cols=3)
    electrode_index = levels[0].tolist().index(electrode)
    loc = list(range(electrode_index * len(levels[1]) * len(levels[2]),
                     (electrode_index + 1) * len(levels[1]) * len(levels[2])))
    data = trf.iloc[loc]
    trf_matrix = np.reshape(data.values, (len(levels[1]), len(levels[2]))).transpose()
    parameters = trf_parameters.iloc[electrode_index]
    clean_trf = cleaned_trf[electrode_index]
    filter_trf = filtered_trf[electrode_index]
    add_trf_fig_for_electrode(parameters, levels, clean_trf, filter_trf, trf_fig, trf_matrix)
    trf_fig.update_layout(coloraxis=dict(colorscale='Haline'), showlegend=False, margin=dict(l=10, r=10, b=10, t=10))
    trf_fig.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1), yaxis2=dict(scaleanchor="x", scaleratio=1),
                          yaxis3=dict(scaleanchor="x", scaleratio=1), template="plotly_dark")
    trf_fig.update_yaxes(showgrid=False, zeroline=False, linecolor='black', showticklabels=False, ticks='')
    trf_fig.update_xaxes(showgrid=False, zeroline=False, linecolor='black', showticklabels=False, ticks='')
    return trf_fig


def all_channels_trf(trf, list_electrodes):
    levels = trf.index.levels
    nb_cols = 4
    nb_rows = math.ceil(max(levels[0]) / nb_cols)
    index = 0
    all_channel_fig = make_subplots(rows=nb_rows, cols=nb_cols,
                                    shared_xaxes=False, shared_yaxes=False,
                                    x_title='Frequency (Hz)', y_title='Intensity (dB)')
    for electrode in list_electrodes:
        data = trf.iloc[
            list(range(index * len(levels[1]) * len(levels[2]), (index + 1) * len(levels[1]) * len(levels[2])))]
        trf_matrix = np.reshape(data.values, (len(levels[1]), len(levels[2]))).transpose()

        all_channel_trf_map = go.Heatmap(z=trf_matrix, coloraxis="coloraxis", name="Electrode_" + str(electrode))
        row = electrode % nb_rows
        column = math.floor(electrode / nb_rows)
        all_channel_fig.add_trace(all_channel_trf_map, row=row + 1, col=column + 1)
        index += 1
    all_channel_fig.update_layout(coloraxis=dict(colorscale='Haline'), showlegend=False, template="plotly_dark")
    all_channel_fig.update_yaxes(showgrid=False, zeroline=False, linecolor='black', showticklabels=False, ticks='')
    all_channel_fig.update_xaxes(showgrid=False, zeroline=False, linecolor='black', showticklabels=False, ticks='')
    return all_channel_fig


def add_trf_fig_for_electrode(trf_parameters, levels, cleaned_trf, filtered_trf, trf_fig, trf_matrix):
    trf_map = go.Heatmap(z=trf_matrix, coloraxis="coloraxis", name="TRF_map")
    filtered_trf_map = go.Heatmap(z=filtered_trf, coloraxis="coloraxis", name="Filtered_TRF_map")
    cleaned_trf_map = go.Heatmap(z=cleaned_trf, coloraxis="coloraxis", name="Cleaned_TRF_map")
    trf_fig.add_trace(trf_map, row=1, col=1)
    trf_fig.add_trace(filtered_trf_map, row=1, col=2)
    trf_fig.add_trace(cleaned_trf_map, row=1, col=3)
    index_BF = levels[1].tolist().index(trf_parameters["BF"])
    index_THR = levels[2].tolist().index(trf_parameters["THR"])
    index_CF = levels[1].tolist().index(trf_parameters["CF"])
    index_THRCF = levels[2].tolist().index(trf_parameters["ThrCF"])
    if index_BF is not None and index_THR is not None:
        trf_fig.add_shape(
            type="line", xref="x", yref="y", x0=index_BF, y0=index_THR, x1=index_BF, y1=len(levels[2]) - 1,
            line=dict(color="Red", width=3),
            row=1, col=3
        )
    if index_CF is not None and index_THRCF is not None:
        trf_fig.add_shape(
            type="circle", xref="x", yref="y", x0=index_CF - 0.5, y0=index_THRCF - 0.5, x1=index_CF + 0.5,
            y1=index_THRCF + 0.5,
            line=dict(color="Red", width=1),
            row=1, col=3
        )


def waveform_plot(units, electrode, unit_index=None):
    fs = 24414.0625 / 1000
    x = (np.array(range(101)) - 50) / fs
    x_rev = x[::-1]
    start_process_time = time.process_time()
    fig = go.Figure()
    all_units = units.to_dataframe()
    electrodes = list(all_units["electrodes"].values)
    electrode_unit = [index for index in range(len(electrodes)) if
                      len(electrodes[index].index.values) > 0 and
                      (electrodes[index]["imp"].index.values[0] == electrode).any()]

    if len(electrode_unit) > 0:
        waveform_mean = all_units.iloc[electrode_unit[0]].waveform_mean
        waveform_std = all_units.iloc[electrode_unit[0]].waveform_sd.flatten()
        y_upper = np.add(waveform_mean, waveform_std).tolist()
        y_lower = np.subtract(waveform_mean, waveform_std).tolist()

        mean_fig = go.Scatter(x=x, y=waveform_mean, mode="lines", showlegend=False)
        y1 = list(y_upper) + list(y_lower[::-1])
        x1 = list(x) + list(x_rev)
        std_fig = go.Scatter(
            x=x1,
            y=y1,
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line_color='rgba(255,255,255,0)',
            showlegend=False
        )
        fig.add_trace(mean_fig)
        fig.add_trace(std_fig)
    fig.update_layout(margin=dict(l=10, r=10, b=10, t=10), template="plotly_dark")
    print("Waveform Plot - elapsed time ", (time.process_time() - start_process_time))

    return fig


if __name__ == '__main__':
    path = "C:/Users/jujud/Documents/Consulting/Data/191128EM/NWB"
    nwbfiles = {"Block " + (file.split('-')[-1].replace('.nwb', '')).zfill(2): os.path.join(
        "C:/Users/jujud/Documents/Consulting/Data/191128EM/NWB", file) for file in os.listdir(path) if ".nwb" in file}
    c_file = list(nwbfiles.keys())[0]
    nwb_io = NWBHDF5IO(nwbfiles[c_file], 'r')
    nwbfile = nwb_io.read()
    labbook = LabBookLoader().load_notebook("C:/Users/jujud/Documents/Consulting/Data/", "Labbook_191128EM.xlsx")
    Dashboard.create(nwbfile, nwbfiles, labbook)
    app.run_server(debug=False)
