import numpy as np
from plotly import graph_objects as go
from src.bsl_python.GUI.visualizations.visualization import Visualization


class Raster(Visualization):
    all_spikes = None
    electrode = None
    height = 0.38

    def __init__(self, all_spikes, electrode):
        super(Raster, self).__init__(height=38, name="raster")
        self.electrode = electrode
        self.all_spikes = all_spikes

    def create_figure(self):
        return raster_plot(self.all_spikes, self.electrode)


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
    y_step = np.min(np.diff(np.unique(electrode_spikes[initial_variable].tolist()))) * 0.9 if len(
        electrode_spikes) > 0 else 0
    y = electrode_spikes[initial_variable].tolist() + repetition / np.max(repetition) * y_step - y_step / 2 if len(
        electrode_spikes) > 0 else []
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