import os
import pandas
import dash

import dash_html_components as html
import dash_bootstrap_components as dbc
from pynwb import NWBHDF5IO
import dash_table

from src.bsl_python.GUI.experiments.experiment_factory import ExperimentFactory
from src.bsl_python.GUI.panels.left_panel import LeftPanel
from src.bsl_python.GUI.panels.panel import Panel
from src.bsl_python.GUI.visualizations.all_channels_trf import all_channels_trf
from src.bsl_python.lab_book_loader import LabBookLoader
from src.bsl_python.GUI.app import app


class Dashboard:
    __instance = None
    nwb_files = []
    c_file = None
    nwb_file = None
    notebook = None
    experiment = None
    electrode = 0
    panels = []

    @staticmethod
    def get_instance():
        """ Static access method. """
        if Dashboard.__instance == None:
            Dashboard()
        return Dashboard.__instance

    def __init__(self):
        """ Virtually private constructor. """
        if Dashboard.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            Dashboard.__instance = self

    def create(self, nwb_files, notebook, c_file):
        self.nwb_files = nwb_files
        self.notebook = notebook
        self.c_file = c_file

        file_list = []
        block_names = sorted(list(nwb_files.keys()))

        for index in range(len(nwb_files)):
            file_list.append({'label': notebook['Trials']['StimulusSet'][index] + " - " + block_names[index],
                              'value': block_names[index]})
        left_panel = LeftPanel(file_list, self.c_file)
        middle_panel = Panel(width=6, name="middle_panel")
        right_panel = Panel(width=4, name="right_panel")
        self.panels = [left_panel, middle_panel, right_panel]
        nwb_io = NWBHDF5IO(nwb_files[list(nwb_files.keys())[c_file]], 'r')
        nwb_file = nwb_io.read()
        self.experiment = ExperimentFactory.create_experiment(nwb_file.protocol, nwb_file, self.panels)
        app.layout = html.Div(id="dashboard", children=[dbc.Row(children=self.experiment.get_html(), className="m-1"),
                                                        dbc.Toast(
                                                            [html.P("This is the content of the toast",
                                                                    className="mb-0")],
                                                            id="warning-toast",
                                                            header="Warning",
                                                            icon="danger",
                                                            dismissable=True,
                                                            is_open=False,
                                                            style={"position": "fixed", "top": 10, "right": 10,
                                                                   "width": 350})
                                                        ])


@app.callback(
    [
        dash.dependencies.Output('changeable_left_panel', 'children'),
        dash.dependencies.Output('middle_panel', 'children'),
        dash.dependencies.Output('right_panel', 'children'),
        dash.dependencies.Output('warning-toast', 'is_open'),
        dash.dependencies.Output('warning-toast', 'children')],
    [dash.dependencies.Input('electrode-dropdown', 'value'), dash.dependencies.Input('file-dropdown', 'value')],
    prevent_initial_call=True)
def change_electrode(electrode, file):
    changed_dashboard = False
    dashboard = Dashboard.get_instance()
    error = None
    if file != list(dashboard.nwb_files.keys())[dashboard.c_file]:
        dashboard.c_file = list(dashboard.nwb_files.keys()).index(file)
        nwb_io = NWBHDF5IO(dashboard.nwb_files[file], 'r')
        nwb_file = nwb_io.read()
        try:
            dashboard.experiment = ExperimentFactory.create_experiment(nwb_file.protocol, nwb_file, dashboard.panels)
        except NameError as e:
            error = e

        changed_dashboard = True
    if electrode != dashboard.electrode:
        dashboard.electrode = electrode
        dashboard.experiment.electrode = electrode
        changed_dashboard = True
    if changed_dashboard:
        dashboard.panels[0].empty_panel()
        dashboard.panels[1].empty_panel()
        dashboard.panels[2].empty_panel()
        dashboard.experiment.create_visualizations()
    if error is not None:
        return dash.no_update, dash.no_update, dash.no_update, True, [html.P(error, className="mb-0")]
    return dashboard.panels[0].get_html_changeable_part(), dashboard.panels[1].create_children_html(), \
           dashboard.panels[2].create_children_html(), dash.no_update, dash.no_update


if __name__ == '__main__':
    path = "C:/Users/jujud/Documents/Consulting/Data/191128EM/NWB"
    nwb_files = {"Block " + (file.split('-')[-1].replace('.nwb', '')).zfill(2): os.path.join(
        "C:/Users/jujud/Documents/Consulting/Data/191128EM/NWB", file) for file in os.listdir(path) if ".nwb" in file}
    labbook = LabBookLoader().load_notebook("C:/Users/jujud/Documents/Consulting/Data/", "Labbook_191128EM.xlsx")
    Dashboard.get_instance().create(nwb_files, labbook, 0)
    app.run_server(debug=False)
