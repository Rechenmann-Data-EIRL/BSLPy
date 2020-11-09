import numpy as np
from plotly import graph_objects as go

from src.bsl_python.GUI.visualizations.visualization import Visualization


# Peristimulus/poststimulus time histogram
class PSTH(Visualization):
    all_spikes = None
    mean_filtered_activity_rate = None
    electrode = None

    def __init__(self, all_spikes, mean_filtered_activity_rate, electrode):
        super(PSTH, self).__init__(height=20, name="psth")
        self.electrode = electrode
        self.mean_filtered_activity_rate = mean_filtered_activity_rate
        self.all_spikes = all_spikes

    def create_figure(self):
        return psth_plot(self.all_spikes, self.mean_filtered_activity_rate, self.electrode)


def psth_plot(all_spikes, mean_filtered_activity_rate, electrode):
    new_fig = go.Figure()
    electrode_spikes = all_spikes[all_spikes["electrodes"] == electrode]
    spikes = electrode_spikes.trial_time.tolist()
    psth_fig = go.Histogram(x=spikes, xbins=dict(start=-0.2, end=3.5, size=(3.5 + 0.2) / 500), name="activity rate")
    average_activity_fig = go.Scatter(x=np.linspace(-0.2, 1, int(1.2 / 0.001)),
                                      y=mean_filtered_activity_rate[electrode],
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