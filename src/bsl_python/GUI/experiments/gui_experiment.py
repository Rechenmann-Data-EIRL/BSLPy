from abc import ABC, abstractmethod


class GUIExperiment(ABC):
    panels = []
    visualizations = []
    experiment = None

    def __init__(self, experiment):
        self.experiment = experiment
        self.create_panels()
        self.create_visualizations()

    @abstractmethod
    def create_panels(self):
        pass

    @abstractmethod
    def create_visualizations(self):
        pass

    @abstractmethod
    def get_html(self):
        pass
