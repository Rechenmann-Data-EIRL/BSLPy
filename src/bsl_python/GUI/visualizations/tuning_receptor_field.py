import numpy as np
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from src.bsl_python.GUI.visualizations.visualization import Visualization


class TuningReceptorField(Visualization):
    trf_processor = None
    electrode = None
    unit_index = None

    def __init__(self, trf_processor, electrode, unit_index=None):
        super(TuningReceptorField, self).__init__(height=20, name="trf")
        self.electrode = electrode
        self.trf_processor = trf_processor
        self.unit_index = unit_index

    def create_figure(self):
        trf = self.trf_processor.trf
        cleaned_trf = self.trf_processor.cleaned_trf
        filtered_trf = self.trf_processor.filtered_trf
        trf_parameters = self.trf_processor.parameters
        return trf_plot(trf, cleaned_trf, filtered_trf, trf_parameters, self.electrode)


def trf_plot(trf, cleaned_trf, filtered_trf, trf_parameters, electrode):
    levels = trf.index.levels
    trf_fig = make_subplots(rows=1, cols=3, shared_yaxes=True, )
    electrode_index = levels[0].tolist().index(electrode)
    loc = list(range(electrode_index * len(levels[1]) * len(levels[2]),
                     (electrode_index + 1) * len(levels[1]) * len(levels[2])))
    data = trf.iloc[loc]
    trf_matrix = np.reshape(data.values, (len(levels[1]), len(levels[2])))
    parameters = {key: trf_parameters[key][electrode_index] for key in trf_parameters.keys()}
    clean_trf = cleaned_trf[electrode_index]
    filter_trf = filtered_trf[electrode_index]
    add_trf_fig_for_electrode(clean_trf, filter_trf, trf_fig, trf_matrix)
    draw_parameters(levels, trf_fig, parameters)

    freq_level = levels[1].tolist()
    decb_level = levels[2].tolist()
    tick_x_values = list(range(0, len(decb_level), int(np.ceil(len(decb_level)/5)))),
    tick_x_text = [str(round(decb_level[tick])) for tick in tick_x_values[0]],
    tick_y_values = list(range(0, len(freq_level), int(np.ceil(len(freq_level)/5)))),
    tick_y_text = [str(round(freq_level[tick])) for tick in tick_y_values[0]],
    for column in range(1, 4):
        trf_fig.update_xaxes(showgrid=False, zeroline=False, linecolor='black', ticktext=tick_x_text[0], tickvals=tick_x_values[0],
                             row=1, col=column)
    trf_fig.update_xaxes(title=levels[2].name, row=1, col=2)
    trf_fig.update_yaxes(title=levels[1].name, showgrid=False, zeroline=False, linecolor='black', ticktext=tick_y_text[0], tickvals=tick_y_values[0], row=1,
                         col=1)
    trf_fig.update_layout(coloraxis=dict(colorscale='Haline'), showlegend=False, margin=dict(l=10, r=10, b=15, t=10))
    trf_fig.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1), template="plotly_dark")
    return trf_fig


def add_trf_fig_for_electrode(cleaned_trf, filtered_trf, trf_fig, trf_matrix):
    trf_map = go.Heatmap(z=trf_matrix, coloraxis="coloraxis", name="TRF_map")
    filtered_trf_map = go.Heatmap(z=filtered_trf, coloraxis="coloraxis", name="Filtered_TRF_map")
    cleaned_trf_map = go.Heatmap(z=cleaned_trf, coloraxis="coloraxis", name="Cleaned_TRF_map")
    trf_fig.add_trace(trf_map, row=1, col=1)
    trf_fig.add_trace(filtered_trf_map, row=1, col=2)
    trf_fig.add_trace(cleaned_trf_map, row=1, col=3)


def draw_parameters(levels, trf_fig, trf_parameters):
    if not np.isnan(trf_parameters["BF"]) and not np.isnan(trf_parameters["THR"]):
        index_BF = levels[2].tolist().index(trf_parameters["BF"])
        index_THR = levels[1].tolist().index(trf_parameters["THR"])
        index_CF = levels[2].tolist().index(trf_parameters["CF"])
        index_THRCF = levels[1].tolist().index(trf_parameters["ThrCF"])
        if index_BF is not None and index_THR is not None:
            trf_fig.add_shape(
                type="line", xref="x", yref="y", x0=index_BF, y0=index_THR, x1=index_BF, y1=len(levels[1]) - 1,
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
