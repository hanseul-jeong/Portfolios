import random

class ReplayMemory(object):

    def __init__(self, start_index, end_index, batch_size):
        """
        :param start_index: start index of the training set on the global data matrices
        :param end_index: end index of the training set on the global data matrices
        """
        self.__experiences = [i for i in range(start_index, end_index)]
        # NOTE: in order to achieve the previous w feature
        self.__batch_size = batch_size

    def push(self, state_index):
        self.__experiences.append(state_index)

    def __sample(self, start, end):
        """
        @:param end: is excluded
        @:param bias: value in (0, 1)
        """
        ran = random.randint(end-start+1)
        result = end - ran
        return result

    def next_batch(self):
        batch_start = self.__sample(0, len(self.__experiences) - self.__batch_size)
        return batch_start

