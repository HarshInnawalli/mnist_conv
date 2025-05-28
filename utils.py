import matplotlib.pyplot as plt

def display_prediction(image, predicted, true):
    plt.imshow(image.reshape(28, 28), cmap='gray')
    plt.title(f'Predicted: {predicted}, True: {true}')
    plt.axis('off')
    plt.show()
