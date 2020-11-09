from abc import ABC, abstractmethod
import dash_core_components as dcc


class Visualization(ABC):

    def __init__(self, height, name):
        self.height = height
        self.default_height = height
        self.name = name

    @abstractmethod
    def create_figure(self):
        pass

    def get_html(self):
        return dcc.Graph(
            id=self.name,
            figure=self.create_figure(),
            style={"height": str(self.height) + "vh"}
        )