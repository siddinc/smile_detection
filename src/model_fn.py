from datetime import datetime
from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D, BatchNormalization, Flatten
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utility_fn import load_dataset, preprocess_dataset, plot_metrics
from constants import BATCH_SIZE, SAVE_MODEL_PATH, LOAD_MODEL_PATH, MAPPING


def create_model(input_shape, classes):
    i = Input(shape=input_shape)

    x = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(i)
    x = BatchNormalization()(x)

    x = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D(pool_size=(4, 4))(x)

    x = Conv2D(128, (3, 3), strides=(1, 1), activation='relu')(x)
    x = BatchNormalization()(x)

    x = Conv2D(128, (3, 3), strides=(1, 1), activation='relu')(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D(pool_size=(8, 8))(x)

    x = Flatten()(x)
    x = Dense(100, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(classes, activation='softmax')(x)

    model = Model(i, x)
    return model


def compile_model(model, optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']):
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)


def training_data_augmentation(x_train, y_train, batch_size):
    data_generator = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.1,
        horizontal_flip=True,
    )

    train_generator = data_generator.flow(
        x_train, y_train, batch_size=batch_size)
    steps_per_epoch = x_train.shape[0] // BATCH_SIZE
    return (train_generator, steps_per_epoch)


def train_model(model, train_generator, x_test, y_test, no_of_epochs, steps_per_epoch):
    r = model.fit_generator(
        train_generator,
        validation_data=(x_test, y_test),
        epochs=no_of_epochs, steps_per_epoch=steps_per_epoch, verbose=1
    )
    return r


def evaluate_model(model, x_test, y_test, batch_size):
    loss, accuracy = model.evaluate(x_test, y_test, batch_size=batch_size)
    print(loss, accuracy)


def save_model(model):
    now = datetime.now()
    model_name_suffix = now.strftime('%d-%m-%Y-%H:%M:%S')
    save_model(model, SAVE_MODEL_PATH + '/model${}.h5'.format(model_name_suffix))


def load_saved_model(model_name):
    loaded_model = load_model(LOAD_MODEL_PATH + '/{}'.format(model_name))
    return loaded_model


def predict_image(model, image):
    predicted_labels = model.predict(image)
    result_label = MAPPING[predicted_labels.argmax(axis=1)[0]]
    score = max(predicted_labels[0]) * 100
    return (str(result_label), float(score))