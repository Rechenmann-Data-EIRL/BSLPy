from abc import ABC, abstractmethod

from src.bsl_python.GUI.experiment_info import ExperimentInfo
from src.bsl_python.preprocessing.experiments.experiment_factory import ExperimentFactory


class GUIExperiment(ABC):
    panels = []
    visualizations = []
    experiment = None
    experiment_info = None

    def __init__(self, nwb_file, panels):
        self.experiment = ExperimentFactory.create_experiment(nwb_file.protocol, nwb_file)
        self.experiment_info = ExperimentInfo(nwb_file)
        self.panels = panels
        print("Pre-process")
        self.experiment.preprocess()
        print("Create Visualizations")
        self.create_visualizations()

    @abstractmethod
    def create_visualizations(self):
        pass

    @abstractmethod
    def get_html(self):
        pass

