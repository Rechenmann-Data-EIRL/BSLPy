import math
import pandas as pd
from hdmf.common.table import DynamicTable
from pynwb import ProcessingModule
from src.bsl_python.preprocessing.processor.processor import Processor
import numpy as np


class Parameters(Processor):
    activity_parameters = None

    def __init__(self, spiking_activity, mean_filtered_activity):
        super(Parameters, self).__init__('parameters', 'All extracted parameters')
        spontaneous_activity = get_activity(spiking_activity, -0.15, -0.05)
        post_stim_activity = get_activity(spiking_activity, 0.3, 0.5)
        mean_spontaneous_activity = {electrode: np.mean(spontaneous_activity[electrode]) for electrode in
                                     spiking_activity.keys()}
        std_spontaneous_activity = {electrode: np.std(spontaneous_activity[electrode]) for electrode in
                                    spiking_activity.keys()}
        mean_post_stim_activity = {electrode: np.mean(post_stim_activity[electrode]) for electrode in
                                   spiking_activity.keys()}
        std_post_stim_activity = {electrode: np.std(post_stim_activity[electrode]) for electrode in
                                  spiking_activity.keys()}
        peak_amplitude = {electrode: np.max(mean_filtered_activity[electrode]) for electrode in
                          mean_filtered_activity.keys()}
        peak_latency = {
            electrode: (np.where(mean_filtered_activity[electrode] == peak_amplitude[electrode])[0][0] - 200) * 0.001
            for electrode in mean_filtered_activity.keys()}
        self.activity_parameters = pd.DataFrame.from_dict({"electrode": list(spiking_activity.keys()),
                                                           "mean_spontaneous_activity": list(
                                                               mean_spontaneous_activity.values()),
                                                           "activity_peak_amplitude": list(peak_amplitude.values()),
                                                           "activity_peak_latency": list(peak_latency.values()),
                                                           "std_spontaneous_activity": list(
                                                               std_spontaneous_activity.values()),
                                                           "mean_post_stim": list(mean_post_stim_activity.values()),
                                                           "std_post_stim": list(std_post_stim_activity.values())})

    def create_module(self):
        module = ProcessingModule(name=self.name, description=self.description)
        module.add_container(DynamicTable.from_dataframe(self.activity_parameters, name="activity_parameters"))
        return module


def get_activity(activity, tmin, tmax):
    index_tmin = math.floor((tmin + 0.2) / 0.001)
    index_tmax = math.floor((tmax + 0.2) / 0.001)
    new_activity = {electrode: [activity[electrode][trial][index_tmin:index_tmax] for trial in
                                range(len(activity[electrode]))] for electrode in activity.keys()}
    return new_activity
