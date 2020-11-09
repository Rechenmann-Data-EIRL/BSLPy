from src.bsl_python.GUI.experiment_info import ExperimentInfo
from src.bsl_python.GUI.panels.panel import Panel
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import dash_table


class LeftPanel(Panel):
    files = None
    file_index = None
    electrode = None
    electrodes = None

    def __init__(self, files, file_index):
        super(LeftPanel, self).__init__(width=2, name="left_panel")
        self.files = files
        self.file_index = file_index

    def get_html(self):
        return dbc.Col(children=[
            dbc.Row(children=[
                dbc.Col(children=[
                    "Blocks",
                    dcc.Dropdown(
                        id='file-dropdown',
                        options=self.files,
                        value=self.files[self.file_index]["value"],
                        style={"color": "black"},
                        className="mb-1"
                    ),
                    "Electrodes",
                    dcc.Dropdown(
                        id='electrode-dropdown',
                        options=[{'label': str(electrode), 'value': electrode} for electrode in self.electrodes],
                        value=self.electrode,
                        style={"color": "black"},
                        className="mb-2"
                    ),
                    html.Hr(style={"border": "1px dashed white"})],
                    width={"size": 12})]),
            dbc.Row(id="changeable_left_panel", children=self.get_html_changeable_part())
        ], width={"size": 2})

    def get_html_changeable_part(self):
        return [dbc.Col(children=[child.get_html() for child in self.children], width={"size": 12})]


