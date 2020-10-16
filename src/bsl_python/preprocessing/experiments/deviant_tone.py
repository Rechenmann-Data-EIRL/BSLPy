from src.bsl_python.preprocessing.experiments.experiment import Experiment
from src.bsl_python.preprocessing.experiments.experiment import Experiment
from src.bsl_python.preprocessing.processor.parameters import Parameters
from src.bsl_python.preprocessing.processor.spiking_activity import FilteredActivity, DefaultSpikingActivity, \
    MeanFilteredActivity, MeanActivity
from src.bsl_python.preprocessing.processor.stimulation_activity import StimulationActivity
from src.bsl_python.preprocessing.processor.tuning_receptor_field import TuningReceptorField
from src.bsl_python.preprocessing.processor.waveform import Waveform


class DeviantTone(Experiment):
    def __init__(self, spikes):
        super(DeviantTone, self).__init__(spikes)

    def get_stimuli_conditions(self):
        self.stimuli_conditions = [{'name': 'Level, dB',
                                    'key': 'decB',
                                    'value': self.spikes.decB}]
        return self.stimuli_conditions

    def compute_processing_time_window(self):
        pass

    def preprocess(self):
        fs = 24414.0625 / 1000
        list_trials = range(len(self.info))
        #list_electrodes = self.channels
        list_electrodes = [60, 61, 62, 63]
        activity = DefaultSpikingActivity(fs, list_electrodes, self.spikes, list_trials)
        filtered_activity = FilteredActivity(activity)
        mean_activity = MeanActivity(activity, list_trials)
        mean_filtered_activity = MeanFilteredActivity(filtered_activity, list_trials)
        parameters = Parameters(activity.activity, mean_filtered_activity.activity)
        waveform = Waveform(self.units)
        self.processors.append(activity)
        self.processors.append(filtered_activity)
        self.processors.append(mean_activity)
        self.processors.append(mean_filtered_activity)
        self.processors.append(parameters)
        self.processors.append(waveform)
