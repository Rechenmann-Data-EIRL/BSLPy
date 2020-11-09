import time

import numpy as np
from plotly import graph_objects as go
from src.bsl_python.GUI.visualizations.visualization import Visualization


class Waveform(Visualization):
    units = None
    electrode = None
    unit_index = None

    def __init__(self, units, electrode, unit_index=None):
        super(Waveform, self).__init__(height=20, name="waveform")
        self.electrode = electrode
        self.units = units
        self.unit_index = unit_index

    def create_figure(self):
        return waveform_plot(self.units, self.electrode, self.unit_index)


def waveform_plot(units, electrode, unit_index=None):
    fs = 24414.0625 / 1000
    x = (np.array(range(101)) - 50) / fs
    x_rev = x[::-1]
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
    return fig
