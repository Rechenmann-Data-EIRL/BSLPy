from src.bsl_python.GUI.experiments.deviant_tone import DeviantTone
from src.bsl_python.GUI.experiments.tone_fine import ToneFine
from src.bsl_python.GUI.experiments.fms import FMS

class ExperimentFactory:

    @staticmethod
    def create_experiment(name, nwb_file, left_panel):
        if name == 'DeviantTone':
            experiment = DeviantTone(nwb_file, left_panel)
        elif name == 'ToneFine':
            experiment = ToneFine(nwb_file, left_panel)
        elif name == 'ToneCoarse':
            experiment = ToneFine(nwb_file, left_panel)
        elif name == 'FMS':
            experiment = FMS(nwb_file, left_panel)
        else:
            raise NameError('Experiment ' + name + ' is not listed as implemented experiment.')
        return experiment
