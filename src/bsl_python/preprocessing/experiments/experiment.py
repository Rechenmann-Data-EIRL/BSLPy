import math
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
        spikes = np.where(np.logical_and(start_time - 0.2 < spike_times, spike_times < start_time + 3.5))
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
    def __init__(self, nwb_file):
        self.channels = nwb_file.electrodes
        self.spikes, self.info, self.units = build_dataframe(nwb_file)
        self.stimuli_conditions = []
        self.processors = []

    @abstractmethod
    def get_stimuli_conditions(self):
        pass

    def compute_processing_time_window(self):
        pass

    @abstractmethod
    def preprocess(self):
        pass

    def compute_sweep_time(self):
        fq_min = 2000
        fq_max = 48000
        sweep_oct = abs(math.log2(fq_max / fq_min))
        sweepTime = abs(sweep_oct / len(self.stimuli_conditions[1])) / 1000

    def save(self, nwb_file):
        if "spikes" in nwb_file.processing:
            nwb_file.processing.pop("spikes")
        spikes_module = ProcessingModule(name='spikes', description='All extracted spikes')
        spikes_module.add_container(DynamicTable.from_dataframe(self.spikes, name="spikes"))
        nwb_file.add_processing_module(spikes_module)
        for processor in self.processors:
            processor.replace_module(nwb_file)


