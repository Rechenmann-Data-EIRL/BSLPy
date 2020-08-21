import os

import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from pynwb import NWBHDF5IO

from src.bsl_python.GUI.nwb_loader import NWBFileLoader, parse_contents, UPLOAD_DIRECTORY
from src.bsl_python.GUI.app import app


class ExperimentInfo:
    @staticmethod
    def get_html(file):
        if file is None:
            return ExperimentInfo.get_empty_html()
        return html.Div(id="experiment-info-panel",
                        children=[dbc.Label("Lab: " + file.lab), html.Br(),
                                  dbc.Label("Institution: " + file.institution), html.Br(),
                                  dbc.Label("Experimenter: " + file.experimenter[0]), html.Br(),
                                  dbc.Label("Experiment ID: " + file.session_id), html.Br(),
                                  dbc.Label("Protocol: " + file.protocol)])

    @staticmethod
    def get_empty_html():
        html.Div(id="experiment-info-panel", children=[dbc.Label("Lab: -"), html.Br(),
                                                       dbc.Label("Institution: -"), html.Br(),
                                                       dbc.Label("Experimenter: -"), html.Br(),
                                                       dbc.Label("Experiment ID: -"), html.Br(),
                                                       dbc.Label("Protocol: -")])
        pass


class Index:
    def __init__(self, files=None):
        self.title = "Welcome to Brain Sound Lab"
        self.files = files

    def get_html(self, file=None):
        button_upload_id = 'upload-data'

        @app.callback(Output('main-container', 'children'), [Input(button_upload_id, 'contents')],
                      [State(button_upload_id, 'filename'), State(button_upload_id, 'last_modified')],
                      prevent_initial_call=True)
        def load_nwb_files(list_of_contents, list_of_names, list_of_dates):
            if list_of_contents is not None:
                page = Index([{"value": parse_contents(c, n), "filename": n} for c, n, d in
                              zip(list_of_contents, list_of_names, list_of_dates)])
                return page.get_html()

        if self.files is None:
            return self.create_welcome_page(button_upload_id)
        else:
            @app.callback([Output('dashboard', 'children'), Output("experiment-info-panel", 'children')],
                          [Input('select-block', 'value'), Input('select-block', 'options')])
            def load_nwb_file(value):
                nwb_io = NWBHDF5IO(os.path.join(UPLOAD_DIRECTORY, value), 'r')
                nwbfile = nwb_io.read()
                return html.Div(), ExperimentInfo.get_html(nwbfile)
            self.files.sort(key=select_lowest)
            return html.Div(id="main-container", children=[
                dbc.Row(
                    dbc.Col(
                        html.H2(children='Welcome to Brain Sound Lab', className="text-center"),
                        width={"size": 6}, align="center"
                    ), justify="center", align="center"
                ),
                dbc.Row(children=[
                    dbc.Col(
                        children=[
                            NWBFileLoader().get_button(button_upload_id),
                            dbc.Label("Select Block"),
                            dbc.Select(id="select-block",
                                options=[{"label": "Block " + str(child["value"].identifier) + " - " + child["value"].protocol,
                                          "value": child["filename"]} for child
                                         in self.files]),
                            html.Br(),
                            ExperimentInfo().get_empty_html()
                        ], width={"size": 3}),
                    dbc.Col(children=[html.Div(id="dashboard")])
                ])
            ])


    def create_welcome_page(self, button_upload_id):
        return dbc.Container(id="main-container", children=[
            dbc.Row(
                dbc.Col(
                    children=[
                        html.H2(children='Welcome to Brain Sound Lab', className="text-center"),
                        NWBFileLoader().get_selector(button_upload_id)],
                    width={"size": 6}, align="center"
                ), justify="center", align="center"
            ),
        ])


def select_lowest(nwb_file):
    return int(nwb_file["value"].identifier)
