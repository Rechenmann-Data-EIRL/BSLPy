import math

import pandas as pd
import scipy.signal
from hdmf.common.table import DynamicTable
from pynwb import ProcessingModule
import numpy as np


class TuningCurve:
    def compute(self, experiment, stimulation_levels, spikesTimeProc, spontaneous_spikes, list_electrodes):
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
        for electrode in list_electrodes:
            fq_min = 2000
            fq_max = 48000
            sweep_oct = abs(math.log2(fq_max / fq_min))
            sweepTime = abs(sweep_oct / len(stimulation_levels(2))) / 1000
