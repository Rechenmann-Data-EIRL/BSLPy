from hdmf.common.table import DynamicTable
from pynwb import ProcessingModule

from src.bsl_python.preprocessing.processor.processor import Processor
import functools
import operator
from math import floor
import numpy as np
import pandas as pd

class StimulationActivity(Processor):
    activity = None

    def __init__(self, activity, experiment_info, feature_1_name, feature_2_name, list_trials):
        super(StimulationActivity, self).__init__('stimulation_activity',
                                                  'Organized average activity per electrode and trial')
        self.compute(activity, experiment_info, feature_1_name, feature_2_name, list_trials)

    def compute(self, activity, experiment_info, feature_1_name, feature_2_name, list_trials):
        stimulation_activity = functools.reduce(operator.iconcat, [
            np.sum(get_activity_for_electrode(activity.activity, electrode, 0.01, 0.06, np.max(list_trials) + 1),
                   1) * 0.001 / 0.05 for electrode in activity.activity.keys()], [])
        stimulation_param1 = functools.reduce(operator.iconcat,
                                              [[row[feature_1_name] for row in experiment_info] for electrode in
                                               activity.activity.keys()], [])
        stimulation_param2 = functools.reduce(operator.iconcat,
                                              [[row[feature_2_name] for row in experiment_info] for electrode in
                                               activity.activity.keys()], [])
        electrodes = functools.reduce(operator.iconcat,
                                      [[electrode] * len(experiment_info) for electrode in activity.activity.keys()],
                                      [])
        self.activity = {'activity': stimulation_activity,
                         feature_1_name: stimulation_param1,
                         feature_2_name: stimulation_param2,
                         'electrode': electrodes}
        return self.activity

    def create_module(self):
        module = ProcessingModule(name=self.name, description=self.description)
        module.add_container(DynamicTable.from_dataframe(pd.Dataframe(self.activity), name="stimulation_activtiy"))
        return module


def get_activity_for_electrode(activity, electrode, tmin, tmax, list_trials):
    if tmin is None or tmax is None:
        return [[0]] * list_trials
    index_tmin = floor((tmin + 0.2) / 0.001)
    index_tmax = floor((tmax + 0.2) / 0.001)
    new_activity = [activity[electrode][trial][index_tmin:index_tmax] for trial in
                    range(len(activity[electrode]))]
    return new_activity
