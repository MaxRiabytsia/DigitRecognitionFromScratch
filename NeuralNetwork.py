import numpy as np
from typing import *


class NeuralNetwork:
    def __init__(self, input_layer_neurons_number: int,
                 hidden_layer_neurons_number: int, output_layer_neurons_number: int):
        self.input_layer_neurons_number = input_layer_neurons_number
        self.hidden_layer_neurons_number = hidden_layer_neurons_number
        self.output_layer_neurons_number = output_layer_neurons_number

        self.hidden_layer_weights = np.random.uniform(-0.5, 0.5, size=(self.hidden_layer_neurons_number,
                                                                       self.input_layer_neurons_number))
        self.hidden_layer_biases = np.random.uniform(-0.5, 0.5, size=(self.hidden_layer_neurons_number, 1))

        self.output_layer_weights = np.random.uniform(-0.5, 0.5, size=(self.output_layer_neurons_number,
                                                                       self.hidden_layer_neurons_number))
        self.output_layer_biases = np.random.uniform(-0.5, 0.5, size=(self.output_layer_neurons_number, 1))

    def train(self, train_data: np.array, train_labels: np.array, epochs: int, learning_rate: float) -> None:
        train_data, train_labels = NeuralNetwork.__unison_shuffle(train_data, train_labels)

        for epoch in range(1, epochs + 1):
            self.__forward_propagation(train_data)
            self.__backward_propagation(train_data, train_labels)
            self.__update_weights_and_biases(learning_rate)

            if epoch % 10 == 0:
                accuracy = NeuralNetwork.__get_accuracy(self.output_layer_output, train_labels)
                print(f"================ Epoch {epoch} ================\n"
                      f"Training Accuracy: {round(accuracy * 100, 2)}% \n")

    def __forward_propagation(self, train_data: np.array):
        self.hidden_layer_input = self.hidden_layer_weights.dot(train_data.T) + self.hidden_layer_biases
        self.hidden_layer_output = NeuralNetwork.__relu(self.hidden_layer_input)
        self.output_layer_input = self.output_layer_weights.dot(self.hidden_layer_output) + self.output_layer_biases
        self.output_layer_output = NeuralNetwork.__softmax(self.output_layer_input)

    def __backward_propagation(self, train_data: np.array, train_labels: np.array):
        m = train_labels.size
        one_hot_train_labels = NeuralNetwork.__to_one_hot_encoded(train_labels)
        dZ2 = self.output_layer_output - one_hot_train_labels
        self.dW2 = 1 / m * dZ2.dot(self.hidden_layer_output.T)
        self.dB2 = 1 / m * np.sum(dZ2)
        dZ1 = self.output_layer_weights.T.dot(dZ2) * NeuralNetwork.__relu_derivative(self.hidden_layer_input)
        self.dW1 = 1 / m * dZ1.dot(train_data)
        self.dB1 = 1 / m * np.sum(dZ1)

    def __update_weights_and_biases(self, learning_rate: float):
        self.hidden_layer_weights -= learning_rate * self.dW1
        self.hidden_layer_biases -= learning_rate * self.dB1
        self.output_layer_weights -= learning_rate * self.dW2
        self.output_layer_biases -= learning_rate * self.dB2

    def evaluate(self, test_data: np.array, test_labels: np.array):
        self.__forward_propagation(test_data)
        accuracy = NeuralNetwork.__get_accuracy(self.output_layer_output, test_labels)
        print(f"Testing Accuracy: {round(accuracy * 100, 2)}% \n")

    def predict(self, datum: np.array):
        self.__forward_propagation(datum.reshape(1, -1))
        prediction = NeuralNetwork.__get_predictions(self.output_layer_output)
        return prediction[0], self.output_layer_output[prediction][0][0]

    @staticmethod
    def __get_predictions(output: np.array) -> np.array:
        return np.argmax(output, 0)

    @staticmethod
    def __get_accuracy(predictions: np.array, labels: np.array) -> float:
        predictions = NeuralNetwork.__get_predictions(predictions)
        return np.sum(predictions == labels) / labels.size

    @staticmethod
    def __relu(x: Union[float, np.array]) -> Union[float, np.array]:
        return np.maximum(x, 0)

    @staticmethod
    def __relu_derivative(x) -> np.array:
        return x > 0

    @staticmethod
    def __softmax(x: Union[float, np.array]) -> Union[float, np.array]:
        return np.exp(x) / sum(np.exp(x))

    @staticmethod
    def __to_one_hot_encoded(labels: np.array) -> np.array:
        result = np.zeros((labels.shape[0], np.max(labels) + 1))
        result[np.arange(labels.shape[0]), labels] = 1
        return result.T

    @staticmethod
    def __unison_shuffle(array1: np.array, array2: np.array) -> tuple[np.array, np.array]:
        assert len(array1) == len(array2)
        p = np.random.permutation(len(array1))
        array1, array2 = array1[p], array2[p]
        return array1, array2
