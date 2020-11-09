import math

import pandas as pd
import scipy.signal
from hdmf.common.table import DynamicTable
from pynwb import ProcessingModule
import numpy as np

from src.bsl_python.preprocessing.processor.processor import Processor


class TuningCurve(Processor):
    stimulation_spikes = None
    spontaneous_spikes = None
    stimulation_levels = None
    electrodes = None
    curve_parameters = None
    spontaneous_window = None

    def __init__(self, stimulation_spikes, spontaneous_spikes, stimulation_levels, electrodes, repetition, spontaneous_window):
        super(TuningCurve, self).__init__("tuning_curve", "")
        self.stimulation_spikes = stimulation_spikes
        self.spontaneous_spikes = spontaneous_spikes
        self.repetition = repetition
        self.stimulation_levels = stimulation_levels
        self.electrodes = electrodes
        self.spontaneous_window = spontaneous_window
        self.compute()

    def compute(self):
        parameters = {"TRFnum": [],
                      "TRFnum_max": [],
                      "TRFrate": [],
                      "SpontTRFrate": [],
                      "stdSpontTRFrate": [],
                      "TRFrate_max": [],
                      "peakLatency_stim1": [],
                      "peakAmplitude_stim1": [],
                      "onsetLatency_stim1": [],
                      "peakLatency_stim2": [],
                      "peakAmplitude_stim2": [],
                      "onsetLatency_stim2": []
                      }
        feature_1_key = self.stimulation_levels[0]["key"]
        feature_1 = self.stimulation_levels[0]["value"]
        feature_2_key = self.stimulation_levels[1]["key"]
        feature_2 = self.stimulation_levels[1]["value"]
        selected_spikes = self.stimulation_spikes.groupby(["electrodes", feature_2_key, feature_1_key])
        selected_spont_spikes = self.spontaneous_spikes.groupby(["electrodes", feature_2_key, feature_1_key])
        counted_spikes = selected_spikes.count()["in_processing_range"]
        counted_spont_spikes = selected_spont_spikes.count()["trial_time"]
        trf_shape = len(self.electrodes)*len(np.unique(feature_1))*len(np.unique(feature_2))
        values = {"trf": [0]*trf_shape, 'spont_trf': [0]*trf_shape}
        repetitions = np.reshape([self.repetition.values for electrode in self.electrodes], trf_shape)
        levels = pd.MultiIndex.from_product([-self.electrodes.to_dataframe()["imp"].values, np.unique(feature_2), np.unique(feature_1)],
                                            names=['electrode', feature_2_key, feature_1_key])
        self.curve_parameters = pd.DataFrame(values, index=levels)
        trf_values = counted_spikes.values / selected_spikes["sweep_time"].mean()
        spont_trf_values = counted_spont_spikes.values / (self.spontaneous_window['max'] - self.spontaneous_window['min'])
        self.curve_parameters.loc[counted_spikes.index, ["trf"]] = trf_values
        self.curve_parameters.loc[counted_spont_spikes.index, ["spont_trf"]] = spont_trf_values
        self.curve_parameters["trf"] /= repetitions
        self.curve_parameters["spont_trf"] /= repetitions

    def create_module(self):
        module = ProcessingModule(name=self.name, description=self.description)
        module.add_container(DynamicTable.from_dataframe(self.curve_parameters, name="curve_parameters"))
        return module
