import numpy as np
import random

class DataLoader:
    def __init__(self, data, batch_size, shuffle=True):
        # Pairing X and Y together in a list of tuples
        self.data = list(zip(*data))
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.current_index = 0
        if shuffle:
            self._shuffle_data()

    def __iter__(self):
        self.current_index = 0
        if self.shuffle:
            self._shuffle_data()
        return self

    def __next__(self):
        raise NotImplementedError("Subclasses should implement this!")

    def _shuffle_data(self):
        random.shuffle(self.data)

class Batch_Loader(DataLoader):
    def __init__(self, data, shuffle=True):
        super().__init__(data, len(data[0]), shuffle)

    def __next__(self):
        if self.current_index >= len(self.data):
            raise StopIteration
        self.current_index = len(self.data)  # Return full dataset once
        # Convert to numpy arrays
        X_batch = np.array([x for x, y in self.data])
        Y_batch = np.array([y for x, y in self.data])
        return X_batch, Y_batch

class MiniBatch_Loader(DataLoader):
    def __init__(self, data, batch_size=32, shuffle=True):
        super().__init__(data, batch_size, shuffle)

    def __next__(self):
        if self.current_index >= len(self.data):
            raise StopIteration
        mini_batch = self.data[self.current_index:self.current_index + self.batch_size]
        self.current_index += self.batch_size
        # Unpack mini_batch into separate X and Y batches and convert to numpy arrays
        X_batch = np.array([x for x, y in mini_batch])
        Y_batch = np.array([y for x, y in mini_batch])
        return X_batch, Y_batch

class SimpleLoader(MiniBatch_Loader):
    def __init__(self, data, shuffle=False):
        # Set batch_size to 1 for single data point iteration
        super().__init__(data, batch_size=1, shuffle=shuffle)

if __name__ == "__main__":
    X = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    Y = np.array([1, 1, 0, 0])

    print("Testing SimpleLoader:")
    simple_loader = SimpleLoader((X, Y), shuffle=False)
    for x, y in simple_loader:
        print("X:", x, "Y:", y)

    print("\nTesting MiniBatch_Loader with batch size of 2:")
    mini_batch_loader = MiniBatch_Loader((X, Y), batch_size=2, shuffle=False)
    for X_batch, Y_batch in mini_batch_loader:
        print("X batch:", X_batch, "Y batch:", Y_batch)

    print("\nTesting Batch_Loader (entire dataset at once):")
    batch_loader = Batch_Loader((X, Y), shuffle=False)
    for X_full, Y_full in batch_loader:
        print("X full batch:", X_full, "Y full batch:", Y_full)
