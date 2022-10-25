import cv2
import numpy as np
import os

from NeuralNetwork import NeuralNetwork


def get_data() -> tuple[np.array, np.array, np.array, np.array]:
    test_size = 0.3
    train_images, train_labels, test_images, test_labels = [], [], [], []
    for folder in os.listdir("data/trainingSet"):
        images = os.listdir(f"data/trainingSet/{folder}")
        for i, image_name in enumerate(images[:1000]):
            image_path = f"data/trainingSet/{folder}/{image_name}"
            pixels = cv2.imread(image_path, 0).flatten()
            if i < len(images) * test_size:
                train_images.append(pixels)
                train_labels.append(folder)
            else:
                test_images.append(pixels)
                test_labels.append(folder)

    return np.array(train_images), np.array(train_labels), np.array(test_images), np.array(test_labels)


def get_neural_net(train_images: np.array, train_labels: np.array) -> NeuralNetwork:
    neural_net = NeuralNetwork(
        input_layer_neurons_number=train_images.shape[1],
        hidden_layer_neurons_number=32,
        output_layer_neurons_number=10
    )
    neural_net.train(train_images, train_labels, epochs=10, batch_size=128, learning_rate=0.05)
    return neural_net


def evaluate_neural_network(neural_net: NeuralNetwork, test_images: np.array, test_labels: np.array) -> None:
    if not neural_net:
        print("Cannot evaluate until the neural network is trained.")
        return None
    neural_net.evaluate(test_images, test_labels)


def recognize_digit_from_img(neural_net: NeuralNetwork) -> None:
    if not neural_net:
        print("Cannot recognize digits until the neural network is trained.")
        return None

    # img_path = input("Enter image path: ")
    img_path = "data/trainingSample/0/img_1.jpg"
    pixels = cv2.imread(img_path, 0)
    cv2.imshow('Image', cv2.resize(cv2.imread(img_path, 0), (256, 256)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    digit, probability = neural_net.predict(pixels.flattened)
    print(f"The neural network is {probability * 100}% sure that the digit on the image is '{digit}'")


def main():
    print("Рябиця Максим Андрійович\n"
          "ІПЗ-21\n")

    neural_net = None
    train_images, train_labels, test_images, test_labels = get_data()
    while True:
        print("\n1. Train neural network\n"
              "2. Evaluate neural network\n"
              "3. Recognize digit from image\n"
              "0. Exit\n")

        task_number = int(input("Enter task number: "))
        match task_number:
            case 1:
                neural_net = get_neural_net(train_images, train_labels)
            case 2:
                evaluate_neural_network(neural_net, test_images, test_labels)
            case 3:
                recognize_digit_from_img(neural_net)
            case 0:
                break


if __name__ == "__main__":
    main()
