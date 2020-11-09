from abc import abstractmethod

import pynwb
from pynwb import ProcessingModule

from src.bsl_python.preprocessing.processor.processor import Processor
import numpy as np
from scipy import signal


class SpikingActivity(Processor):
    frequency = None
    electrodes = None
    activity = None

    # len(experiment.info)
    def __init__(self, name, description, frequency, electrodes):
        super(SpikingActivity, self).__init__(name, description)
        self.frequency = frequency
        self.electrodes = electrodes

    @abstractmethod
    def compute(self):
        pass

    def create_module(self):
        module = ProcessingModule(name=self.name, description=self.description)
        for electrode in self.electrodes:
            module.add(
                pynwb.base.TimeSeries('electrode' + str(electrode), self.activity[electrode], unit="spikes/s",
                                      rate=self.frequency,
                                      comments="Each row corresponds to a specific trial"))
        return module


class DefaultSpikingActivity(SpikingActivity):
    spikes = None
    trials = None

    def __init__(self, frequency, electrodes, spikes, trials):
        super(DefaultSpikingActivity, self).__init__('spiking_activity', 'Spiking activity per trial and electrode',
                                                     frequency, electrodes)
        self.spikes = spikes
        self.trials = trials
        self.compute()

    def compute(self):
        t_max = 1
        t_min = -0.2
        n_bins = round((t_max - t_min) / 0.001)
        self.activity = dict()

        for electrode in self.electrodes:
            electrode_spikes = self.spikes[self.spikes.electrodes.eq(electrode)]
            self.activity[electrode] = np.array([
                np.histogram(electrode_spikes['trial_time'][electrode_spikes.trials.eq(trial)], n_bins,
                             range=(t_min, t_max))[0] * 1000 for trial in self.trials])


class FilteredActivity(SpikingActivity):
    spiking_activity = None

    def __init__(self, spiking_activity):
        super(FilteredActivity, self).__init__('filtered_activity',
                                               'Filtered spiking activity with hanning window per trial and electrode',
                                               spiking_activity.frequency, spiking_activity.electrodes)
        self.spiking_activity = spiking_activity
        self.compute()

    def compute(self):
        # A beta value of 14 is probably a good starting point. Note that as beta gets large, the window narrows,
        # and so the number of samples needs to be large enough to sample the increasingly narrow spike, otherwise NaNs
        # will get returned. Most references to the Kaiser window come from the signal processing literature, where it is
        # used as one of many windowing functions for smoothing values. It is also known as an apodization (which means
        # “removing the foot”, i.e. smoothing discontinuities at the beginning and end of the sampled signal) or tapering
        # function.
        window = np.hanning(9)  # window length of 15 points and a beta of 14
        window = window / np.sum(window)
        self.activity = {
            electrode: signal.filtfilt(window, 1, self.spiking_activity.activity[electrode], axis=1,
                                       method="gust").tolist()
            for electrode in self.spiking_activity.activity.keys()}


class MeanFilteredActivity(SpikingActivity):
    filtered_activity = None

    def __init__(self, filtered_activity, trials):
        super(MeanFilteredActivity, self).__init__('mean_filtered_activity',
                                                   'Average filtered spiking activity per electrode',
                                                   filtered_activity.frequency, filtered_activity.electrodes)
        self.filtered_activity = filtered_activity
        self.trials = trials
        self.compute()

    def compute(self):
        self.activity = {
            electrode: np.mean(np.array([self.filtered_activity.activity[electrode][trial] for trial in self.trials]),
                               axis=0) for
            electrode in self.filtered_activity.activity.keys()}


class MeanActivity(SpikingActivity):
    spiking_activity = None
    trials = None

    def __init__(self, spiking_activity, trials):
        super(MeanActivity, self).__init__('mean_spiking_activity', 'Average spiking activity per electrode',
                                           spiking_activity.frequency, spiking_activity.electrodes)
        self.spiking_activity = spiking_activity
        self.trials = trials
        self.compute()

    def compute(self):
        self.activity = {
            electrode: np.mean(np.array([self.spiking_activity.activity[electrode][trial] for trial in self.trials]),
                               axis=0) for
            electrode in self.spiking_activity.activity.keys()}
