import math

import pandas as pd
import scipy.signal
from hdmf.common.table import DynamicTable
from pynwb import ProcessingModule
import numpy as np

from src.bsl_python.preprocessing.processor.processor import Processor


class TuningReceptorField(Processor):
    trf = None
    filtered_trf = None
    cleaned_trf = None
    parameters = None
    filter_size = None
    list_electrodes = None

    def __init__(self, filter_size, stim_activity, spontaneous_rate_mean, peak_amplitude, list_electrodes,
                 feature_1_name, feature_2_name):
        super(TuningReceptorField, self).__init__('trf', 'Tuning Receptor Field')
        self.filter_size = filter_size
        self.compute(stim_activity, spontaneous_rate_mean, peak_amplitude, list_electrodes, feature_1_name,
                     feature_2_name)

    def compute(self, stim_activity, spontaneous_rate_mean, peak_amplitude, list_electrodes, feature_1_name,
                feature_2_name):
        self.list_electrodes = list_electrodes
        df = pd.DataFrame(stim_activity)
        self.trf = df.groupby(['electrode', feature_1_name, feature_2_name]).mean()
        levels = self.trf.index.levels
        self.cleaned_trf = []
        self.filtered_trf = []
        self.parameters = {"dprime": [],
                           "BF": [],
                           "meanFiringRateBF": [],
                           "THR": [],
                           "BW10": [],
                           "CF": [],
                           "ThrCF": [],
                           "BWat60": [],
                           "electrode": []}
        # Step size in octaves
        octave_size = round(np.log2(levels[1][-1] / levels[1][0]) / (len(levels[1]) - 1), 4)
        index = 0
        for electrode in list_electrodes:
            electrode_index = levels[0].tolist().index(electrode)
            loc = list(range(electrode_index * len(levels[1]) * len(levels[2]),
                             (electrode_index + 1) * len(levels[1]) * len(levels[2])))
            data = self.trf.iloc[loc]
            trf_matrix = np.reshape(data.values, (len(levels[1]), len(levels[2])))
            tmp_filtered_trf = scipy.ndimage.median_filter(trf_matrix, (self.filter_size, self.filter_size),
                                                           mode='reflect')
            tmp_cleaned_trf = tmp_filtered_trf - spontaneous_rate_mean[index]
            tmp_cleaned_trf[tmp_cleaned_trf < peak_amplitude[index] * 0.2] = 0
            self.cleaned_trf.append(tmp_cleaned_trf)
            self.filtered_trf.append(tmp_filtered_trf)
            trf_signal = trf_matrix[tmp_cleaned_trf != 0]
            trf_noise = trf_matrix[tmp_cleaned_trf == 0]
            is_empty = len(trf_signal)
            self.parameters["dprime"].append((np.mean(trf_signal) - np.mean(trf_noise)) /
                                             math.sqrt(
                                                 (np.var(trf_signal) + np.var(
                                                     trf_noise)) / 2) if is_empty > 0 else np.nan)
            summed_cleaned_trf_freq = np.sum(tmp_cleaned_trf, 0).tolist()
            summed_cleaned_trf_intensity = np.sum(tmp_cleaned_trf, 1).tolist()
            index_bf = summed_cleaned_trf_freq.index(max(summed_cleaned_trf_freq))
            index_thr = next((index for index in range(len(tmp_cleaned_trf[:, index_bf])) if
                              tmp_cleaned_trf[index, index_bf] > 0), None)
            bf = levels[2][index_bf] if is_empty > 0 else np.nan
            thr = levels[1][index_thr] if is_empty > 0 else np.nan
            bw10 = np.nan
            if not np.isnan(thr) and thr + 10 in levels[1]:
                index_thr_10 = levels[1].tolist().index(thr + 10)
                index_thr_10_min = next(
                    (index_bf - index for index in range(len(tmp_cleaned_trf[index_thr_10, 0:index_bf - 1])) if
                     tmp_cleaned_trf[index_thr_10, index_bf - 1 - index] == 0), None)
                index_thr_10_max = next(
                    (index + index_bf for index in range(len(tmp_cleaned_trf[index_thr_10, index_bf + 1:])) if
                     tmp_cleaned_trf[index_thr_10, index_bf + 1 + index] == 0), None)
                bw10 = (index_thr_10_max - index_thr_10_min) * octave_size
            index_thr_cf = next((index for index in range(len(summed_cleaned_trf_intensity)) if
                                 summed_cleaned_trf_intensity[index] > 0), None)
            thr_cf = levels[1][index_thr_cf] if index_thr_cf is not None else None
            index_cf = np.min(np.where(tmp_cleaned_trf[index_thr_cf, :] == np.max(tmp_cleaned_trf[index_thr_cf, :])), 1)
            cf = levels[2][index_cf[0]] if len(index_cf) > 0 else np.nan

            bwat60 = np.nan
            if 60 in levels[1] and not np.isnan(bf):
                ind60 = np.where(levels[1] == 60)[0][0]
                if index_bf == 0:
                    val60min = np.where(tmp_cleaned_trf[ind60, index_bf] == 0)
                else:
                    val60min = np.where(tmp_cleaned_trf[ind60, 0:(index_bf - 1 if index_bf > 0 else 0)] == 0)[0] + 1
                if len(val60min) == 0:
                    val60min = 0
                else:
                    val60min = val60min[-1]

                val60max = np.where(tmp_cleaned_trf[ind60, index_bf + 1:] == 0)[0][0] - 1 + index_bf
                bwat60 = (val60max - val60min) * octave_size

            self.parameters["BF"].append(bf)
            self.parameters["meanFiringRateBF"].append(np.mean(tmp_cleaned_trf[:, index_bf]))
            self.parameters["THR"].append(thr)
            self.parameters["BW10"].append(bw10)
            self.parameters["ThrCF"].append(thr_cf)
            self.parameters["CF"].append(cf)
            self.parameters["BWat60"].append(bwat60)
            self.parameters["electrode"].append(electrode)
            index += 1

    def create_module(self):
        parameters = pd.DataFrame(self.parameters)
        processed_trf = pd.DataFrame(
            {"filtered_trf": self.filtered_trf, "cleaned_trf": self.cleaned_trf, "electrodes": self.list_electrodes})
        trf_module = ProcessingModule(name=self.name, description=self.description)
        trf_module.add(DynamicTable.from_dataframe(self.trf.reset_index(), name=self.name))
        trf_module.add(
            DynamicTable.from_dataframe(parameters.set_index("electrode").reset_index(), name="trf_parameters"))
        trf_module.add(DynamicTable.from_dataframe(processed_trf.reset_index(), name="processed_trf"))
        return trf_module
