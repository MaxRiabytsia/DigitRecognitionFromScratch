import numpy as np
from typing import *


class NeuralNetwork:
    def __init__(self, input_layer_neurons_number: int,
                 hidden_layer_neurons_number: int, output_layer_neurons_number: int):
        self.input_layer_neurons_number = input_layer_neurons_number
        self.hidden_layer_neurons_number = hidden_layer_neurons_number
        self.output_layer_neurons_number = output_layer_neurons_number

        self.hidden_layer_weights = np.random.uniform(0, 1, size=(self.hidden_layer_neurons_number,
                                                                  self.input_layer_neurons_number))
        self.hidden_layer_biases = np.random.uniform(0, 1, size=self.hidden_layer_neurons_number).reshape(-1, 1)

        self.output_layer_weights = np.random.uniform(0, 1, size=(self.output_layer_neurons_number,
                                                                  self.hidden_layer_neurons_number))
        self.output_layer_biases = np.random.uniform(0, 1, size=self.output_layer_neurons_number).reshape(-1, 1)

    def train(self, train_data: np.array, train_labels: np.array,
              epochs: int, batch_size: int, learning_rate: float) -> None:
        for epoch in range(epochs):
            train_data, train_labels = unison_shuffle(train_data, train_labels)
            cost = 0
            for data_batch, labels_batch in zip(train_data[::batch_size], train_labels[::batch_size]):
                self.__forward_propagation(data_batch)
                cost += NeuralNetwork.__cost_function(train_labels, self.outputs)
                self.__backward_propagation(data_batch, labels_batch)
                self.__update_weights_and_biases(cost, learning_rate)
            print(f"================ Epoch 1 ================\n"
                  f"Training Accuracy: {cost} \n"
                  f"Testing accuracy: {0} \n")

    def __forward_propagation(self, train_data: np.array):
        self.outputs = []
        for datum in train_data:
            self.hidden_layer_input = np.dot(self.hidden_layer_weights, datum) + self.hidden_layer_biases
            self.hidden_layer_output = NeuralNetwork.__relu(self.hidden_layer_input)

            output_layer_input = np.dot(self.output_layer_weights, self.hidden_layer_output) + self.output_layer_biases
            output_layer_output = NeuralNetwork.__softmax(output_layer_input)
            self.outputs.append(output_layer_output)

    @staticmethod
    def __relu(x: Union[float, np.array]) -> Union[float, np.array]:
        return np.maximum(0, x)

    @staticmethod
    def __relu_derivative(x):
        return x > 0

    @staticmethod
    def __softmax(x: Union[float, np.array]) -> Union[float, np.array]:
        return np.exp(x) / np.sum(np.exp(x))

    def __backward_propagation(self, train_data: np.array, train_labels: np.array):
        train_labels = NeuralNetwork.__to_one_hot_encoded(train_labels)

        cost_partial_derivative_to_output_layer_input = (NeuralNetwork.__relu_derivative(
            self.output_layer_weights * self.hidden_layer_output + self.output_layer_biases) *
                                                         NeuralNetwork.__cost_function_derivative(train_labels,
                                                                                                  self.outputs))

        self.cost_partial_derivative_to_output_weights = (
                self.hidden_layer_output * cost_partial_derivative_to_output_layer_input)
        self.cost_partial_derivative_to_output_biases = (1 * NeuralNetwork.__relu_derivative(
            self.output_layer_weights * self.hidden_layer_output + self.output_layer_biases) *
                                                         NeuralNetwork.__cost_function_derivative(train_labels,
                                                                                                  self.outputs))
        cost_partial_derivative_to_hidden_output = (
                self.output_layer_weights * cost_partial_derivative_to_output_layer_input)

        derivative_of_hidden_layer_input = NeuralNetwork.__relu_derivative(
            self.hidden_layer_weights * train_data + self.hidden_layer_biases)

        self.cost_partial_derivative_to_hidden_weights = (train_data * derivative_of_hidden_layer_input *
                                                          cost_partial_derivative_to_hidden_output)
        self.cost_partial_derivative_to_hidden_biases = (1 * derivative_of_hidden_layer_input *
                                                         cost_partial_derivative_to_hidden_output)

    @staticmethod
    def __to_one_hot_encoded(labels: np.array):
        result = np.zeros((labels.shape[0], np.max(labels) + 1))
        result[:, labels] = 1
        return result

    @staticmethod
    def __cost_function(desired_outputs: np.array, real_outputs: np.array) -> np.array:
        return np.sum((desired_outputs - real_outputs) ** 2)

    @staticmethod
    def __cost_function_derivative(desired_outputs: np.array, real_outputs: np.array) -> np.array:
        return np.sum(2 * (desired_outputs - real_outputs))

    def __update_weights_and_biases(self, cost, learning_rate: float):
        delta_output_layer_weights = self.cost_partial_derivative_to_output_weights * self.cost
        delta_output_layer_biases = self.cost_partial_derivative_to_output_biases * self.cost
        delta_hidden_layer_weights = self.cost_partial_derivative_to_hidden_weights * self.cost
        delta_hidden_layer_biases = self.cost_partial_derivative_to_hidden_biases * self.cost

        self.output_layer_weights += delta_output_layer_weights * learning_rate
        self.output_layer_biases += delta_output_layer_biases * learning_rate
        self.hidden_layer_weights += delta_hidden_layer_weights * learning_rate
        self.hidden_layer_biases += delta_hidden_layer_biases * learning_rate

    def evaluate(self, test_images: np.array, test_labels: np.array):
        pass

    def predict(self, datum: np.array):
        pass


def unison_shuffle(array1: np.array, array2: np.array):
    assert len(array1) == len(array2)
    p = np.random.permutation(len(array1))
    array1, array2 = array1[p], array2[p]
    return array1, array2
