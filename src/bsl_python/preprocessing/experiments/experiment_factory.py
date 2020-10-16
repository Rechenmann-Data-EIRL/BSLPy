from src.bsl_python.preprocessing.experiments.deviant_tone import DeviantTone
from src.bsl_python.preprocessing.experiments.tone_coarse import ToneCoarse
from src.bsl_python.preprocessing.experiments.tone_fine import ToneFine


class ExperimentFactory:

    @staticmethod
    def create_experiment(name, nwb_file):
        if name == 'DeviantTone':
            experiment = DeviantTone(nwb_file)
        elif name == 'ToneFine':
            experiment = ToneFine(nwb_file)
        elif name == 'ToneCoarse':
            experiment = ToneCoarse(nwb_file)
        else:
            raise NameError('Experiment ' + name + ' is not listed as implemented experiment.')
        return experiment
