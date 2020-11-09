from numpy import int64

from src.bsl_python.GUI.experiments.gui_experiment import GUIExperiment
from src.bsl_python.GUI.panels.panel import Panel
from src.bsl_python.GUI.visualizations.all_channels_trf import AllChannelsTRF
from src.bsl_python.GUI.visualizations.html_table import HTMLTable
from src.bsl_python.GUI.visualizations.psth import PSTH
from src.bsl_python.GUI.visualizations.raster import Raster
from src.bsl_python.GUI.visualizations.tuning_receptor_field import TuningReceptorField
from src.bsl_python.GUI.visualizations.waveform import Waveform
from pandas import DataFrame
from src.bsl_python.GUI.app import app


class ToneFine(GUIExperiment):
    units = None
    electrodes = []
    electrode = None

    def __init__(self, nwb_file, left_panel):
        self.units = nwb_file.units
        self.electrodes = -nwb_file.electrodes['imp'].data[()].astype(int64)
        self.electrode = self.electrodes[0]
        super(ToneFine, self).__init__(nwb_file, left_panel)

    def create_visualizations(self):
        activity_parameters = self.experiment.get_processor("parameters").activity_parameters
        mean_filtered_activity = self.experiment.get_processor("mean_filtered_activity").activity
        raster = Raster(self.experiment.spikes, self.electrode)
        psth = PSTH(self.experiment.spikes, mean_filtered_activity, self.electrode)
        trf = TuningReceptorField(self.experiment.get_processor("trf"), self.electrode)
        waveform = Waveform(self.units, self.electrode)
        all_channels_trf = AllChannelsTRF(self.experiment.get_processor("trf"), self.electrodes)
        params = DataFrame.from_dict({"Parameter": activity_parameters.columns,
                                      "Value": activity_parameters.loc[int(self.electrode)].values.round(2)})
        self.panels[1].add_children([raster, psth, trf, waveform])
        self.panels[2].add_child(all_channels_trf)
        self.panels[0].electrodes = self.electrodes
        self.panels[0].electrode = self.electrode
        self.panels[0].add_child(self.experiment_info)
        self.panels[0].add_child(HTMLTable("parameters", params))

    def get_html(self):
        html = [panel.get_html() for panel in self.panels]
        return html
