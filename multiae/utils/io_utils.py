"""
ConfigReader: class for loading a given configuration file

Logger: class for logging dictionary items

ResultsWriter: class for writing results file

"""
import yaml


class ConfigReader:
    def __init__(self, config_path):
        with open(config_path, "r") as conf_file:
            conf = yaml.load(conf_file, yaml.FullLoader)

        self._conf = conf


class Logger:
    def __init__(self):
        self.logs = {}

    def on_train_init(self, keys):
        for k in keys:
            self.logs[k] = []

    def on_step_fi(self, logs_dict):
        for k, v in logs_dict.items():
            self.logs[k].append(v.detach().cpu().numpy())


class ResultsWriter:
    def __init__(self, filepath=None):
        self.filepath = filepath

    def write(self, string):
        if self.filepath is None:
            print(string)
        else:
            with open(self.filepath, "a") as txtfile:
                txtfile.write(string)
