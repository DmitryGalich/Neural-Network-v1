import random


class Input:
    value = 0.0
    weight_to_next_0 = random.random()
    weight_to_next_1 = random.random()

    def get_value(self, num):
        self.value = num


class Hidden:
    error = 0.0
    value = 0.0
    weight_delta = 0.0
    weight_to_next = random.random()

    def get_error(self, output_weight_delta):
        self.error = self.weight_to_next * output_weight_delta

    def get_weight_delta(self, sigmoid_dx_value):
        self.weight_delta = self.error * sigmoid_dx_value


class Output:
    error = 0.0
    value = 0.0
    weight_delta = 0.0

    def get_error(self, expect):
        self.error = self.value - expect

    def get_weight_delta(self, sigmoid_dx_value):
        self.weight_delta = self.error * sigmoid_dx_value
