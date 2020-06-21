import io

import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from src.bsl_python.GUI.app import app
from dash.dependencies import Input, Output, State
import base64
import pandas as pd
from src.bsl_python.lab_book_loader import LabBookLoader


class LabBookOpener:
    def get_html(self):
        return html.Div([
            dbc.Button("Open Lab book (.xls)", id="open"),
            html.Div(id='output-data-upload'),
            dbc.Modal(
                [
                    dbc.ModalHeader("Select File"),
                    dcc.Upload(
                        id="upload-data",
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


@app.callback(
    Output("modal", "is_open"),
    [Input("open", "n_clicks"), Input("close", "n_clicks")],
    [State("modal", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    #information = LabBookLoader().load_notebook_from_fileobj(io.BytesIO(decoded))
    information = LabBookLoader().load_notebook('Labbook_191128EM.xlsx', 'C:/Users/jujud/Documents/Consulting/Data')
    return filename


@app.callback(Output('output-data-upload', 'children'), [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'), State('upload-data', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children
