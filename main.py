from data import load_and_preprocess_data
from model import build_model
from evaluate import evaluate_model
import tensorflow as tf

def main():
    x_train, y_train, x_test, y_test = load_and_preprocess_data()

    model = build_model()
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(
        x_train, y_train,
        epochs=10,
        batch_size=64,
        validation_split=0.1
    )

    final_train_acc = history.history['accuracy'][-1]
    print(f'\nFinal training accuracy: {final_train_acc * 100:.2f}%')

    evaluate_model(model, x_test, y_test)

if __name__ == '__main__':
    main()
