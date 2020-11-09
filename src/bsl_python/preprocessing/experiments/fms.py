from src.bsl_python.preprocessing.experiments.experiment import Experiment
from src.bsl_python.preprocessing.processor.tuning_curve import TuningCurve
from collections import Counter
from numpy import unique
import math
import pandas as pd
import numpy as np


class FMS(Experiment):
    def __init__(self, nwb_file):
        super(FMS, self).__init__(nwb_file)
        self.set_processing_window()

    def set_stimuli_conditions(self):
        stimuli = [(condition["rate"], condition["decB"]) for condition in self.info]
        unique_stimuli = unique(stimuli, axis=0)
        self.stimuli_conditions = [{'name': 'Level, dB',
                                    'key': 'decB',
                                    'value': unique_stimuli[:, 1]},
                                   {'name': 'Sweep rate, oct/s',
                                    'key': 'rate',
                                    'value': unique_stimuli[:, 0]}]

    def set_repetitions(self):
        columns = [stimuli["key"] for stimuli in self.stimuli_conditions]
        self.repetitions = pd.DataFrame(self.info).groupby(columns).count()["start_time"]

    def compute_sweep_time(self):
        fq_min = 2000
        fq_max = 48000
        sweep_oct = abs(math.log2(fq_max / fq_min))
        return abs(sweep_oct / self.stimuli_conditions[1]["value"]) / 1000 + 0.09

    def set_processing_window(self):
        sweep_time = self.compute_sweep_time()
        self.processing_window = {'min': [0.01] * len(sweep_time), 'max': sweep_time + 0.01}

    def preprocess(self):
        tuning_curve = TuningCurve(self.get_stim_spikes(), self.get_spontaneous_spikes(), self.stimuli_conditions,
                                   self.channels, self.repetitions, self.spontaneous_window)

    def get_stim_spikes(self):
        if "in_processing_range" not in self.spikes:
            feature_1_key = self.stimuli_conditions[0]["key"]
            feature_1 = self.stimuli_conditions[0]["value"]
            feature_2_key = self.stimuli_conditions[1]["key"]
            feature_2 = self.stimuli_conditions[1]["value"]
            self.spikes["in_processing_range"] = [False] * len(self.spikes)
            self.spikes["sweep_time"] = [np.nan] * len(self.spikes)
            nb_spikes = 0
            unique_feat_1 = Counter(self.spikes[feature_1_key].values).values()
            unique_feat_2 = Counter(self.spikes[feature_2_key].values).values()
            for condition_index in range(len(feature_2)):
                filter_spikes = (self.spikes[feature_1_key] == feature_1[condition_index]) & (
                            self.spikes[feature_2_key] == feature_2[condition_index])
                nb_spikes += np.sum(filter_spikes)
                filter_spikes = filter_spikes & (
                                     self.processing_window['min'][condition_index] < self.spikes["trial_time"]) & (
                                     self.spikes["trial_time"] <= self.processing_window['max'][condition_index])
                # filter_spikes = (self.processing_window['min'][condition_index] < self.spikes["trial_time"]) & (
                #         self.spikes["trial_time"] <= self.processing_window['max'][condition_index])
                self.spikes.loc[filter_spikes, ["in_processing_range"]] = True
                self.spikes.loc[filter_spikes, ["sweep_time"]] = self.processing_window['max'][condition_index] - 0.01
        return self.spikes.loc[self.spikes["in_processing_range"]]
