class SearchSpace:
    def __init__(self):
        self.items = []

    def add_item(self, item):
        self.items.append(item)

    def sample(self):
        sample = {}
        for item in self.items:
            sample[item.name] = item.sample()
        return sample

    def __repr__(self):
        return f"SearchSpace({self.items})"