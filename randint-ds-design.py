from random import randint

class Store:
    def __init__(self):
        self.storage = {}
        self.length = 0

    def insert(self, value):
        if value in self.storage:
            self.storage[value] = True
            self.length += 1
        return

    def remove(self, value):
        if value in self.storage:
            del self.storage[value]
            self.length -= 1
        return

    def getRandom(self):
        # TODO: return a random value that is already inserted (with equal
        # probability)
        return
