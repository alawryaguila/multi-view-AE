class ConfigError(Exception):
    def __init__(self, caught):
        self.caught = caught

class InputError(Exception):
    def __init__(self, caught):
        self.caught = caught
