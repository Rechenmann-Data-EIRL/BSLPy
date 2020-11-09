from src.bsl_python.GUI.experiments.gui_experiment import GUIExperiment
import dash_bootstrap_components as dbc

from src.bsl_python.GUI.panels.panel import Panel
from src.bsl_python.GUI.visualizations.psth import PSTH
from src.bsl_python.GUI.visualizations.raster import Raster


class FMS(GUIExperiment):
    units = None

    def __init__(self, nwb_file, left_panel):
        self.units = nwb_file.units
        super(FMS, self).__init__(nwb_file, left_panel)

    def create_visualizations(self):
        activity_parameters = self.experiment.get_processor("parameters").activity_parameters
        mean_filtered_activity = self.experiment.get_processor("mean_filtered_activity").activity
        electrode_list = list(activity_parameters.index.values)
        electrode = electrode_list[0]
        raster = Raster(self.experiment.spikes, electrode)
        psth = PSTH(self.experiment.spikes, mean_filtered_activity, electrode)
        self.panels[1].children = [raster.get_html(), psth.get_html()]
        self.panels[0].electrodes = electrode_list
        self.panels[0].electrode = electrode

    def get_html(self):
        return [panel.get_html for panel in self.panels]