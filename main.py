import cv2
import numpy as np
import os

from NeuralNetwork import NeuralNetwork


def apply_noise(pixels: np.array) -> np.array:
    pixels += np.random.uniform(-0.3, 0.3, pixels.shape)
    pixels = (pixels - np.min(pixels)) / (np.max(pixels) - np.min(pixels))
    return pixels


def get_data() -> tuple[np.array, np.array, np.array, np.array, np.array, np.array]:
    test_size = 0.3
    clean_test_size = test_size / 2
    train_images, train_labels, test_images, test_labels, noisy_test_images, noisy_test_lables = [], [], [], [], [], []
    for folder in os.listdir("data/trainingSet"):
        images = os.listdir(f"data/trainingSet/{folder}")
        for i, image_name in enumerate(images):
            image_path = f"data/trainingSet/{folder}/{image_name}"
            pixels = cv2.imread(image_path, 0).flatten() / 255.
            if i > len(images) * test_size:
                if i % 2 == 0:
                    pixels = apply_noise(pixels)
                train_images.append(pixels)
                train_labels.append(int(folder))
            elif len(images) * clean_test_size < i < len(images) * test_size:
                pixels = apply_noise(pixels)
                noisy_test_images.append(pixels)
                noisy_test_lables.append(int(folder))
            else:
                test_images.append(pixels)
                test_labels.append(int(folder))

    return (np.array(train_images), np.array(train_labels), np.array(test_images),
            np.array(test_labels), np.array(noisy_test_images), np.array(noisy_test_lables))


def get_neural_net(train_images: np.array, train_labels: np.array) -> NeuralNetwork:
    neural_net = NeuralNetwork(
        input_layer_neurons_number=train_images.shape[1],
        hidden_layer_neurons_number=32,
        output_layer_neurons_number=10
    )
    neural_net.train(train_images, train_labels, epochs=2000, learning_rate=0.5)
    return neural_net


def evaluate_neural_network(neural_net: NeuralNetwork, test_images: np.array, test_labels: np.array,
                            noisy_test_images: np.array, noisy_test_lables: np.array) -> None:
    if not neural_net:
        print("Cannot evaluate until the neural network is trained.")
        return None
    print("Evaluation results for all images: ")
    neural_net.evaluate(np.concatenate([test_images, noisy_test_images]),
                        np.concatenate([test_labels, noisy_test_lables]))
    print("Evaluation results for clean images: ")
    neural_net.evaluate(test_images, test_labels)
    print("Evaluation results for images with noise: ")
    neural_net.evaluate(noisy_test_images, noisy_test_lables)


def recognize_digit_from_img(neural_net: NeuralNetwork) -> None:
    if not neural_net:
        print("Cannot recognize digits until the neural network is trained.")
        return None

    img_path = input("Enter image path: ")
    img_path = "data/trainingSample/" + img_path
    pixels = cv2.imread(img_path, 0) / 255.

    to_apply_noise = input("Do you want to apply noise to the image? (y/n)\n")
    if to_apply_noise.lower() == 'y':
        pixels = apply_noise(pixels)

    cv2.imshow('Image', cv2.resize(pixels, (256, 256)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    digit, probability = neural_net.predict(pixels.flatten())
    print(f"The neural network is {round(probability * 100, 2)}% sure that the digit on the image is '{digit}'")


def main():
    print("Рябиця Максим Андрійович\n"
          "ІПЗ-21\n")

    neural_net = None
    train_images, train_labels, test_images, test_labels, noisy_test_images, noisy_test_lables = get_data()
    while True:
        print("\n1. Train neural network\n"
              "2. Evaluate neural network\n"
              "3. Recognize digit from image\n"
              "0. Exit\n")

        task_number = int(input("Enter task number: "))
        print()
        match task_number:
            case 1:
                neural_net = get_neural_net(train_images, train_labels)
            case 2:
                evaluate_neural_network(neural_net, test_images, test_labels, noisy_test_images, noisy_test_lables)
            case 3:
                recognize_digit_from_img(neural_net)
            case 0:
                break


if __name__ == "__main__":
    main()
