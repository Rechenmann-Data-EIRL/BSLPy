import functools
import math
import operator
import os

import pynwb
from hdmf.common import DynamicTable
import time
from pynwb import NWBHDF5IO, ProcessingModule
import pandas as pd
import numpy as np
import plotly.io as pio

from src.bsl_python.preprocessing.experiments.experiment_factory import ExperimentFactory


pio.templates.default = "plotly_dark"
pd.options.plotting.backend = "plotly"





def create_module_from_activity(name, description, activity, frequency):
    module = ProcessingModule(name=name, description=description)
    for electrode in activity.keys():
        module.add(
            pynwb.base.TimeSeries('electrode' + str(electrode), activity[electrode], unit="spikes/s", rate=frequency,
                                  comments="Each row corresponds to a specific trial"))
    return module


def preprocess_nwbfile(path, filename):
    path_to_file = os.path.join(path, filename)
    nwb_io = NWBHDF5IO(path_to_file, 'r')
    nwb_file = nwb_io.read()

    experiment = ExperimentFactory.create_experiment(nwb_file.protocol, nwb_file)
    experiment.preprocess()
    experiment.save(nwb_file)

    export_filename = filename.replace('.nwb', '_new.nwb')
    new_path_to_file = os.path.join(path, export_filename)
    with NWBHDF5IO(new_path_to_file, mode='w') as export_io:
        export_io.export(src_io=nwb_io, nwbfile=nwb_file)
    nwb_io.close()
    os.remove(path_to_file)
    os.rename(new_path_to_file, path_to_file)





if __name__ == '__main__':
    filename = "191128EM_Block-1.nwb"
    path = "C:/Users/jujud/Documents/Consulting/Data/191128EM/NWB"
    preprocess_nwbfile(path, filename)
