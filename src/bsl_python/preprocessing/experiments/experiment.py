from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from hdmf.common.table import DynamicTable
from pynwb import ProcessingModule

from src.bsl_python.utils import flatten_dict


def build_dataframe(nwb_file):
    spike_times = []
    unit_indices = []
    electrodes = []
    all_units = nwb_file.units.to_dataframe()
    for unit in all_units.itertuples():
        if len(unit) > 0:
            spikes = unit.spike_times.tolist()
            electrode = unit.electrodes.imp.index.tolist()[0]
            spike_times += spikes
            unit_indices += [unit[0]] * len(spikes)
            electrodes += [electrode] * len(spikes)
    trials = range(len(nwb_file.trials))
    trial_indices = np.array([-1.0] * len(spike_times))
    spike_times_per_trial = np.array([-1.0] * len(spike_times))
    spike_times = np.array(spike_times)
    additional_columns = [column for column in nwb_file.trials.colnames if column not in ["start_time", "stop_time"]]
    additional_data = dict()
    for key in additional_columns:
        additional_data[key] = np.array([None] * len(spike_times))
    info = [flatten_dict(nwb_file.trials[trial_index].to_dict()) for trial_index in trials]
    for trial_index in trials:
        start_time = info[trial_index]["start_time"]
        stop_time = info[trial_index]["stop_time"]
        spikes = np.where(np.logical_and(start_time - 0.2 < spike_times, spike_times < stop_time-0.2))
        trial_indices[spikes] = trial_index
        spike_times_per_trial[spikes] = spike_times[spikes] - start_time
        for column in additional_data.keys():
            additional_data[column][spikes] = info[trial_index][column]
    trial_indices = list(trial_indices)
    spike_times_per_trial = list(spike_times_per_trial)
    for column in additional_data.keys():
        additional_data[column] = list(additional_data[column])
    d = {'spike_times': spike_times, 'electrodes': electrodes, "unit": unit_indices, 'trials': trial_indices,
         'trial_time': spike_times_per_trial, **additional_data}
    all_spikes = pd.DataFrame.from_dict(d)
    all_spikes.dropna(inplace=True)
    return all_spikes, info, all_units


class Experiment(ABC):
    stimuli_conditions = []
    repetitions = []
    processing_window = {'min': 0, 'max': 1}
    spontaneous_window = {'min': -0.15, 'max': -0.05}

    def __init__(self, nwb_file):
        self.channels = nwb_file.electrodes
        self.spikes, self.info, self.units = build_dataframe(nwb_file)
        self.processors = []
        self.set_stimuli_conditions()
        self.set_processing_window()
        self.set_repetitions()

    @abstractmethod
    def set_stimuli_conditions(self):
        pass

    @abstractmethod
    def set_processing_window(self):
        pass

    @abstractmethod
    def set_repetitions(self):
        pass

    def compute_processing_time_window(self):
        pass

    @abstractmethod
    def preprocess(self):
        pass

    def save(self, nwb_file):
        if "spikes" in nwb_file.processing:
            nwb_file.processing.pop("spikes")
        spikes_module = ProcessingModule(name='spikes', description='All extracted spikes')
        spikes_module.add_container(DynamicTable.from_dataframe(self.spikes, name="spikes"))
        nwb_file.add_processing_module(spikes_module)
        for processor in self.processors:
            processor.replace_module(nwb_file)

    def get_spikes_in_window(self, t_min, t_max):
        return self.spikes[(t_min < self.spikes["trial_time"]) & (self.spikes["trial_time"] <= t_max)]

    def get_spontaneous_spikes(self):
        return self.get_spikes_in_window(self.spontaneous_window['min'],  self.spontaneous_window['max'])

    def get_spontaneous_duration(self):
        return -0.5 + 0.15

    def get_stim_spikes(self):
        return self.get_spikes_in_window(self.processing_window['min'],  self.processing_window['max'])


