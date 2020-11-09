import dash_bootstrap_components as dbc
import dash_html_components as html


class ExperimentInfo:
    file = None

    def __init__(self, file):
        self.file = file

    def get_html(self):
        if self.file is None:
            return ExperimentInfo.get_empty_html()
        return html.Div(id="experiment-info-panel",
                        children=[dbc.Label("Lab: " + self.file.lab), html.Br(),
                                  dbc.Label("Institution: " + self.file.institution), html.Br(),
                                  dbc.Label("Experimenter: " + self.file.experimenter[0]), html.Br(),
                                  dbc.Label("Experiment ID: " + self.file.session_id), html.Br(),
                                  dbc.Label("Protocol: " + self.file.protocol)])

    @staticmethod
    def get_empty_html():
        html.Div(id="experiment-info-panel", children=[dbc.Label("Lab: -"), html.Br(),
                                                       dbc.Label("Institution: -"), html.Br(),
                                                       dbc.Label("Experimenter: -"), html.Br(),
                                                       dbc.Label("Experiment ID: -"), html.Br(),
                                                       dbc.Label("Protocol: -")])
        pass