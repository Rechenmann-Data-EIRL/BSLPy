from pynwb import ProcessingModule

from src.bsl_python.preprocessing.processor.processor import Processor
from hdmf.common import DynamicTable
import numpy as np
import pandas as pd


class Waveform(Processor):
    analysis = None
    waveform_analysis = None

    def __init__(self, units):
        super(Waveform, self).__init__('waveform_analysis', 'Waveform analysis')
        self.compute(units)

    def create_module(self):
        waveform_module = ProcessingModule(name=self.name, description=self.description)
        waveform_module.add_container(
            DynamicTable.from_dataframe(self.waveform_analysis.transpose(), name=self.name))
        return waveform_module

    def compute(self, units):
        waveform_analysis = {}
        fs = 24414.0625 / 1000
        for unit_index, unit in units.iterrows():
            waveform = unit.waveform_mean
            waveform_reverse = waveform[::-1]
            min_peak_value = min(waveform)
            min_peak_pos = np.where(waveform == min(waveform))[0][0]

            # baseline is taken on 20 values of the waveform before the peak
            baseline = np.mean(waveform[min_peak_pos - 31:min_peak_pos - 11])
            min_peak_amplitude = min_peak_value - baseline
            second_peak_amp = None
            peak_to_through = None
            peak_ratio = None
            slope = None
            width_half_peak = None
            peak_duration = None
            through_duration = None
            spike_dur = None
            if min_peak_pos < len(waveform) - 1:
                second_max_peak_val = max(waveform[min_peak_pos + 1:-1])
                second_max_peak_pos = np.where(waveform == second_max_peak_val)[0][0]
                second_peak_amp = second_max_peak_val - baseline
                peak_to_through = (second_max_peak_pos - min_peak_pos) * 1 / fs
                peak_ratio = min_peak_amplitude / second_peak_amp
                mid_slope_position = (min_peak_pos + second_max_peak_pos) / 2
                step = 0.5 if (min_peak_pos + second_max_peak_pos) % 2 else 1
                slope = (waveform[int(mid_slope_position + step)] - waveform[int(mid_slope_position - step)]) / (
                            2 * step)
                # DEAL WITH THE WIDTH OF HALF PEAK:
                width_half_peak = signal_width(waveform, waveform_reverse, min_peak_pos, min_peak_value / 2) * 1 / fs
                peak_duration = signal_width(waveform, waveform_reverse, min_peak_pos, 0) * 1 / fs
                through_duration = signal_width(-waveform, -waveform_reverse, second_max_peak_pos, 0) * 1 / fs
                spike_dur = peak_duration + through_duration

            waveform_analysis[unit_index] = {"pb_amp": min_peak_amplitude, "tb_amp": second_peak_amp,
                                             "p2t": peak_to_through, "p_rat": peak_ratio, "slope": slope,
                                             "width_half_peak": width_half_peak,
                                             "peak_dur": peak_duration, "through_dur":
                                                 through_duration, "spike_dur": spike_dur}
        self.waveform_analysis = pd.DataFrame.from_dict(waveform_analysis)


def signal_width(waveform, waveform_reverse, position, threshold):
    pos_rev = len(waveform_reverse) - position
    half_dur = next((index for index in range(len(waveform[position:-1])) if waveform[position + index] > threshold),
                    len(waveform) - position)
    half_dur2 = next(
        (index for index in range(len(waveform_reverse[pos_rev:-1])) if waveform_reverse[pos_rev + index] > threshold),
        len(waveform) - pos_rev)
    return half_dur + half_dur2