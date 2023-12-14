class StringTwoCharIterator:
    def __init__(self, string):
        self.string = string
        self.pos = 0

    def has_next(self):
        return self.pos + 2 <= len(self.string)

    def __next__(self):
        if not self.has_next():
            raise StopIteration

        substring = self.string[self.pos : self.pos + 2]
        self.pos += 2
        return substring

    def __iter__(self):
        return self

    def remove(self):
        raise NotImplementedError("remove method is not supported")
