from abc import ABC, abstractmethod


class Processor(ABC):
    name = ''
    description = ''

    def __init__(self, name, description):
        self.name = name
        self.description = description

    def replace_module(self, nwb_file):
        if self.name in nwb_file.processing:
            print(self.name)
            nwb_file.processing.pop(self.name)
        nwb_file.add_processing_module(self.create_module())

    @abstractmethod
    def create_module(self):
        pass
