import dash_bootstrap_components as dbc
import dash_html_components as html
from src.bsl_python.GUI.lab_book_loader import LabBookOpener


class Index:
    def __init__(self):
        self.title = "Welcome to Brain Sound Lab"
        self.menu = LabBookOpener().get_html()

    def get_html(self):
        return html.Div(children=[
            dbc.Row(
                dbc.Col(
                    html.H1(children='Welcome to Brain Sound Lab'),
                    width={"size": 6, "offset": 3},
                )
            ),
            self.menu])


