import io
import os

import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from pynwb import NWBHDF5IO

from src.bsl_python.GUI.app import app
from dash.dependencies import Input, Output, State
import base64

UPLOAD_DIRECTORY = "/temp"


class LabBookLoader:
    @staticmethod
    def get_button(button_id):
        @app.callback(
            Output("modal", "is_open"),
            [Input("open", "n_clicks"), Input("close", "n_clicks")],
            [State("modal", "is_open")],
        )
        def toggle_modal(n1, n2, is_open):
            if n1 or n2:
                return not is_open
            return is_open

        return html.Div([
            dbc.Button("Load Lab Book", id="open"),
            dbc.Modal(
                [
                    dbc.ModalHeader("Select File"),
                    dcc.Upload(
                        id=button_id,
                        children=html.Div(
                            ["Drag and drop or click to select a file to upload."]
                        ),
                        style={
                            "width": "90%",
                            "height": "60px",
                            "lineHeight": "60px",
                            "borderWidth": "1px",
                            "borderStyle": "dashed",
                            "borderRadius": "5px",
                            "textAlign": "center",
                            "margin": "auto",
                        },
                        multiple=True),
                    dbc.ModalFooter(
                        dbc.Button("Close", id="close", className="ml-auto")
                    )
                ],
                id="modal")
        ])

    def get_selector(self, button_id):
        return html.Div([
             dcc.Upload(
                 id=button_id,
                 children=html.Div(
                     ["Drag and drop or click to select a file to upload."]
                 ),
                 style={
                     "width": "100%",
                     "height": "60px",
                     "lineHeight": "60px",
                     "borderWidth": "1px",
                     "borderStyle": "dashed",
                     "borderRadius": "5px",
                     "textAlign": "center",
                     "margin": "10px",
                 },
                 multiple=True,
             )], className="mt-5")


def parse_contents(contents, filename):
    data = contents.encode("utf8").split(b";base64,")[1]
    if not os.path.exists(UPLOAD_DIRECTORY):
        os.mkdir(UPLOAD_DIRECTORY)
    with open(os.path.join(UPLOAD_DIRECTORY, filename), "wb") as fp:
        fp.write(base64.decodebytes(data))
    nwb_io = NWBHDF5IO(os.path.join(UPLOAD_DIRECTORY, filename), 'r')
    nwbfile = nwb_io.read()
    return nwbfile



