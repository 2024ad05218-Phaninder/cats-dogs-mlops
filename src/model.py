"""
CNN Model Architecture for Cats vs Dogs Classification
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_simple_cnn(input_shape=(224, 224, 3), num_classes=2):
    """
    Create a simple CNN model for binary classification

    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of output classes (2 for binary: cats/dogs)

    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 4
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Global Average Pooling
        layers.GlobalAveragePooling2D(),

        # Fully Connected Layers
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])

    return model


def create_mobilenet_model(input_shape=(224, 224, 3), num_classes=2):
    """
    Create a MobileNetV2 model with transfer learning
    Better for edge devices and faster training

    Args:
        input_shape: Input image shape
        num_classes: Number of output classes

    Returns:
        Compiled Keras model
    """
    # Load pre-trained MobileNetV2
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )

    # Freeze base model layers
    base_model.trainable = False

    # Create new model
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model


def compile_model(model, learning_rate=0.001):
    """
    Compile the model with optimizer and loss function

    Args:
        model: Keras model to compile
        learning_rate: Learning rate for optimizer

    Returns:
        Compiled model
    """
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )

    return model


def get_model_summary(model):
    """Get model architecture summary"""
    model.summary()

    # Calculate total parameters
    total_params = model.count_params()
    print(f"\nTotal Parameters: {total_params:,}")

    # Calculate trainable parameters
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    print(f"Trainable Parameters: {trainable_params:,}")

    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params
    }


if __name__ == "__main__":
    # Test model creation
    print("Creating Simple CNN Model...")
    model = create_simple_cnn()
    model = compile_model(model)
    print("\n" + "="*70)
    get_model_summary(model)

    print("\n" + "="*70)
    print("Creating MobileNetV2 Model...")
    mobilenet_model = create_mobilenet_model()
    mobilenet_model = compile_model(mobilenet_model)
    print("\n" + "="*70)
    get_model_summary(mobilenet_model)
