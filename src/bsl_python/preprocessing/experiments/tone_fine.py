from src.bsl_python.preprocessing.experiments.experiment import Experiment
from src.bsl_python.preprocessing.processor.parameters import Parameters
from src.bsl_python.preprocessing.processor.spiking_activity import FilteredActivity, DefaultSpikingActivity, \
    MeanFilteredActivity, MeanActivity
from src.bsl_python.preprocessing.processor.stimulation_activity import StimulationActivity
from src.bsl_python.preprocessing.processor.tuning_receptor_field import TuningReceptorField
from src.bsl_python.preprocessing.processor.waveform import Waveform
from numpy import unique, int64
import pandas as pd


class ToneFine(Experiment):
    def __init__(self, nwb_file):
        super(ToneFine, self).__init__(nwb_file)

    def set_stimuli_conditions(self):
        self.stimuli_conditions = [
            {'name': 'Level, dB',
             'key': 'decB',
             'values': unique(self.spikes.decB.values)},
            {'name': 'Frequency, kHz',
             'key': 'freq',
             'values': unique(self.spikes.freq.values) / 1000}
        ]

    def set_repetitions(self):
        columns = [stimuli["key"] for stimuli in self.stimuli_conditions]
        self.repetitions = pd.DataFrame(self.info).groupby(columns).count()["start_time"]

    def set_processing_window(self):
        self.processing_window = {'min': 0.01, 'max': 0.06}

    def preprocess(self):
        fs = 24414.0625 / 1000
        list_trials = range(len(self.info))
        list_electrodes = -self.channels['imp'].data[()].astype(int64)
        #list_electrodes = [60, 61, 62, 63]
        activity = DefaultSpikingActivity(fs, list_electrodes, self.spikes, list_trials)
        filtered_activity = FilteredActivity(activity)
        mean_activity = MeanActivity(activity, list_trials)
        mean_filtered_activity = MeanFilteredActivity(filtered_activity, list_trials)
        parameters = Parameters(activity.activity, mean_filtered_activity.activity)
        waveform = Waveform(self.units)

        stim_activity = StimulationActivity(activity, self.info, self.stimuli_conditions[0]["key"],
                                            self.stimuli_conditions[1]["key"], list_trials)
        trf = TuningReceptorField(3, stim_activity.activity,
                                  list(parameters.activity_parameters["mean_spontaneous_activity"].values),
                                  list(parameters.activity_parameters["activity_peak_amplitude"].values),
                                  list_electrodes,
                                  self.stimuli_conditions[0]["key"], self.stimuli_conditions[1]["key"])
        self.processors.append(activity)
        self.processors.append(filtered_activity)
        self.processors.append(mean_activity)
        self.processors.append(mean_filtered_activity)
        self.processors.append(parameters)
        self.processors.append(waveform)
        self.processors.append(trf)
