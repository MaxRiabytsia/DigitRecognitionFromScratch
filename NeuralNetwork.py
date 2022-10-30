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

    def train(self, train_data: np.array, train_labels: np.array,
              epochs: int, batch_size: int, learning_rate: float) -> None:
        self.n = train_data.shape[0]
        train_data, train_labels = unison_shuffle(train_data, train_labels)

        if train_data.shape[0] % batch_size != 0:
            batched_train_data = train_data[:-(train_data.shape[0] % batch_size)].reshape(
                (-1, batch_size, self.input_layer_neurons_number))
            batched_train_labels = train_labels[:-(train_labels.shape[0] % batch_size)].reshape((-1, batch_size))
        else:
            batched_train_data = train_data.reshape((-1, batch_size, self.input_layer_neurons_number))
            batched_train_labels = train_labels.reshape((-1, batch_size))

        for epoch in range(1, epochs + 1):
            predictions = []
            for data_batch, labels_batch in zip(batched_train_data, batched_train_labels):
                self.__forward_propagation(data_batch)
                predictions.append(self.output_layer_output)
                self.__backward_propagation(data_batch, labels_batch)
                self.__update_weights_and_biases(learning_rate)

            if epoch % 10 == 0:
                predictions = np.array(predictions).reshape(-1, 10)
                labels = batched_train_labels.reshape(-1, )
                accuracy = round(NeuralNetwork.__get_accuracy(predictions, labels), 4)
                print(f"================ Epoch {epoch} ================\n"
                      f"Training Accuracy: {accuracy * 100}% \n")

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

    def __forward_propagation(self, train_data: np.array):
        self.hidden_layer_input = self.hidden_layer_weights.dot(train_data.T) + self.hidden_layer_biases
        self.hidden_layer_output = NeuralNetwork.__relu(self.hidden_layer_input)

        self.output_layer_input = self.output_layer_weights.dot(self.hidden_layer_output) + self.output_layer_biases
        self.output_layer_output = NeuralNetwork.__softmax(self.output_layer_input)

    @staticmethod
    def __relu(x: Union[float, np.array]) -> Union[float, np.array]:
        return np.maximum(x, 0)

    @staticmethod
    def __relu_derivative(x):
        return x > 0

    @staticmethod
    def __softmax(x: Union[float, np.array]) -> Union[float, np.array]:
        return np.exp(x) / sum(np.exp(x))
        # e_x = np.exp(x - np.max(x))
        # return e_x / e_x.sum()

    @staticmethod
    def __softmax_derivative(x: Union[float, np.array]) -> Union[float, np.array]:
        sx = NeuralNetwork.__softmax(x)
        res = -np.outer(sx, sx) + np.diag(sx.flatten())
        return res

    def __backward_propagation(self, train_data: np.array, train_labels: np.array):
        # dA_out_to_dZ_out = NeuralNetwork.__softmax_derivative(
        #     self.output_layer_weights @ self.hidden_layer_output + self.output_layer_biases)
        # dC_to_dA_out = NeuralNetwork.__cost_function_derivative(train_labels, self.outputs)
        #
        # self.dC_to_dW_out = self.hidden_layer_output @ dA_out_to_dZ_out.reshape(1, -1) * dC_to_dA_out
        # self.dC_to_dW_out = NeuralNetwork.__relu_derivative(
        #     self.output_layer_weights @ self.hidden_layer_output + self.output_layer_biases) * dC_to_dA_out
        #
        # dC_to_dZ_out = self.output_layer_weights.reshape(-1, 10) @ dA_out_to_dZ_out * dC_to_dA_out
        # dA_hid_to_dZ_hid = NeuralNetwork.__relu_derivative(
        #     self.hidden_layer_weights @ train_data.reshape(784, -1) + self.hidden_layer_biases)
        #
        # self.dC_to_dW_hid = train_data.reshape(-1, 128) @ dA_hid_to_dZ_hid.reshape(128, -1) @ dC_to_dZ_out
        #
        # self.dC_to_dB_hid = dA_hid_to_dZ_hid.reshape(-1, self.hidden_layer_neurons_number) @ dC_to_dZ_out
        a = 1
        # n = train_labels.shape[0]
        train_labels = NeuralNetwork.__to_one_hot_encoded(train_labels)
        n = self.n
        dZ2 = self.output_layer_output - train_labels
        self.dW2 = dZ2.dot(self.hidden_layer_output.T) / n
        self.dB2 = np.sum(dZ2) / n
        dZ1 = self.output_layer_weights.T.dot(dZ2) * NeuralNetwork.__relu_derivative(self.hidden_layer_input)
        self.dW1 = dZ1.dot(train_data) / n
        self.dB1 = np.sum(dZ1) / n

    @staticmethod
    def __to_one_hot_encoded(labels: np.array):
        result = np.zeros((labels.shape[0], np.max(labels) + 1))
        result[np.arange(labels.shape[0]), labels] = 1
        return result.T

    @staticmethod
    def __cost_function(desired_outputs: np.array, real_outputs: np.array) -> np.array:
        return np.sum((desired_outputs - real_outputs) ** 2)

    @staticmethod
    def __cost_function_derivative(desired_outputs: np.array, real_outputs: np.array) -> np.array:
        return np.sum(2 * (desired_outputs - real_outputs))

    def __update_weights_and_biases(self, learning_rate: float):
        self.output_layer_weights -= self.dW2 * learning_rate
        self.output_layer_biases -= self.dB2 * learning_rate
        self.hidden_layer_weights -= self.dW1 * learning_rate
        self.hidden_layer_biases -= self.dB1 * learning_rate

    @staticmethod
    def __get_accuracy(predictions, labels):
        predictions = np.argmax(predictions, 1)
        # labels = np.argmax(labels, 1)
        print(predictions[:10], labels[:10])
        return np.sum(predictions == labels) / labels.shape[0]

    def evaluate(self, test_images: np.array, test_labels: np.array):
        pass

    def predict(self, datum: np.array):
        pass


def unison_shuffle(array1: np.array, array2: np.array):
    assert len(array1) == len(array2)
    p = np.random.permutation(len(array1))
    array1, array2 = array1[p], array2[p]
    return array1, array2
