import random

class SearchSpaceItem:
    def __init__(self, name, options):
        self.name = name
        self.options = options

    def sample(self):
        return random.choice(self.options)

    def __repr__(self):
        return f"{self.name}: {self.options}"

