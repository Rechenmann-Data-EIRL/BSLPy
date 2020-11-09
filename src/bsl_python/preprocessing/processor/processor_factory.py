

class ProcessorFactory:
    def create(self, nwb_file, processor_name):
        if processor_name == "DefaultActivity":
            return DefaultSpikingActivity()
