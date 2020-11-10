import math

import numpy as np
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from src.bsl_python.GUI.visualizations.visualization import Visualization


class AllChannelsTRF(Visualization):
    trf = None
    list_electrodes = []
    height = 0.98

    def __init__(self, trf, list_electrodes):
        super(AllChannelsTRF, self).__init__(height=98, name="all_channels_trf")
        self.trf = trf
        self.list_electrodes = list_electrodes

    def create_figure(self):
        return all_channels_trf(self.trf.trf, self.list_electrodes)


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
        trf_matrix = np.reshape(data.values, (len(levels[1]), len(levels[2])))

        all_channel_trf_map = go.Heatmap(z=trf_matrix, coloraxis="coloraxis", name="Electrode_" + str(electrode))
        row = electrode % nb_rows
        column = math.floor(electrode / nb_rows)
        all_channel_fig.add_trace(all_channel_trf_map, row=row + 1, col=column + 1)
        index += 1
    all_channel_fig.update_layout(coloraxis=dict(colorscale='Haline'), showlegend=False, template="plotly_dark",
                                  margin={'t': 50})
    all_channel_fig.update_yaxes(showgrid=False, zeroline=False, linecolor='black', showticklabels=False, ticks='')
    all_channel_fig.update_xaxes(showgrid=False, zeroline=False, linecolor='black', showticklabels=False, ticks='')
    return all_channel_fig
