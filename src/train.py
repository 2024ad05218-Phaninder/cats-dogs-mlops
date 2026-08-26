"""
Training Script for Cats vs Dogs Classification Model
Includes MLflow tracking, checkpointing, and evaluation
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import argparse
from datetime import datetime
import mlflow
import mlflow.tensorflow
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from model import create_simple_cnn, compile_model, get_model_summary


class DataGenerator:
    """Data generator for loading preprocessed images"""

    def __init__(self, data_path, batch_size=32, image_size=224):
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.image_size = image_size

    def load_images_from_directory(self, directory_path):
        """Load images from directory"""
        images = []
        labels = []

        for label, category in enumerate(['cats', 'dogs']):
            category_path = directory_path / category

            if not category_path.exists():
                print(f"Warning: {category_path} not found")
                continue

            image_files = list(category_path.glob('*.jpg'))
            print(f"Loading {len(image_files)} {category} images...")

            for img_path in image_files:
                try:
                    img = tf.keras.preprocessing.image.load_img(
                        img_path,
                        target_size=(self.image_size, self.image_size)
                    )
                    img_array = tf.keras.preprocessing.image.img_to_array(img)
                    img_array = img_array / 255.0  # Normalize

                    images.append(img_array)
                    labels.append(label)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")

        return np.array(images), np.array(labels)

    def get_datasets(self):
        """Load train, validation, and test datasets"""
        train_path = self.data_path / 'train'
        val_path = self.data_path / 'validation'
        test_path = self.data_path / 'test'

        print("\n" + "="*70)
        print("LOADING DATASETS")
        print("="*70)

        print("\nLoading training data...")
        x_train, y_train = self.load_images_from_directory(train_path)

        print("\nLoading validation data...")
        x_val, y_val = self.load_images_from_directory(val_path)

        print("\nLoading test data...")
        x_test, y_test = self.load_images_from_directory(test_path)

        # Convert labels to one-hot encoding
        y_train = keras.utils.to_categorical(y_train, 2)
        y_val = keras.utils.to_categorical(y_val, 2)
        y_test = keras.utils.to_categorical(y_test, 2)

        print(f"\nTrain: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")

        return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def train_model(config):
    """Main training function with MLflow tracking"""

    # Start MLflow experiment
    mlflow.set_experiment(config['experiment_name'])

    with mlflow.start_run():
        print("\n" + "="*70)
        print("TRAINING BASELINE CNN MODEL")
        print("="*70)

        # Log parameters
        mlflow.log_params({
            'epochs': config['epochs'],
            'batch_size': config['batch_size'],
            'learning_rate': config['learning_rate'],
            'model_type': 'Simple CNN',
            'image_size': 224
        })

        # Load data
        data_gen = DataGenerator(
            config['data_path'],
            batch_size=config['batch_size'],
            image_size=224
        )
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = data_gen.get_datasets()

        # Create model
        print("\n" + "="*70)
        print("CREATING MODEL")
        print("="*70)
        model = create_simple_cnn()
        model = compile_model(model, learning_rate=config['learning_rate'])
        get_model_summary(model)

        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=str(Path(config['models_path']) / 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]

        # Train model
        print("\n" + "="*70)
        print("TRAINING")
        print("="*70)
        history = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            callbacks=callbacks,
            verbose=1
        )

        # Evaluate on test set
        print("\n" + "="*70)
        print("EVALUATION")
        print("="*70)
        test_loss, test_accuracy, test_precision, test_recall = model.evaluate(
            x_test, y_test,
            verbose=0
        )

        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall: {test_recall:.4f}")

        # Log metrics
        mlflow.log_metrics({
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'best_val_accuracy': float(np.max(history.history['val_accuracy'])),
            'final_train_accuracy': float(history.history['accuracy'][-1])
        })

        # Save model
        model_path = Path(config['models_path']) / f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
        model.save(str(model_path))
        print(f"\n✅ Model saved to {model_path}")

        # Generate predictions for confusion matrix
        y_pred_probs = model.predict(x_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_test_labels = np.argmax(y_test, axis=1)

        # Generate classification report
        report = classification_report(
            y_test_labels, y_pred,
            target_names=['cats', 'dogs'],
            output_dict=True
        )

        print("\n" + "="*70)
        print("CLASSIFICATION REPORT")
        print("="*70)
        print(classification_report(
            y_test_labels, y_pred,
            target_names=['cats', 'dogs']
        ))

        # Save metrics
        metrics_file = Path(config['models_path']) / 'metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump({
                'test_accuracy': float(test_accuracy),
                'test_loss': float(test_loss),
                'test_precision': float(test_precision),
                'test_recall': float(test_recall),
                'classification_report': report
            }, f, indent=2)

        # Save confusion matrix
        cm = confusion_matrix(y_test_labels, y_pred)
        cm_file = Path(config['models_path']) / 'confusion_matrix.npy'
        np.save(cm_file, cm)

        # Plot and save confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Cats', 'Dogs'],
                    yticklabels=['Cats', 'Dogs'])
        plt.title('Confusion Matrix - Test Set')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        cm_plot_path = Path(config['models_path']) / 'confusion_matrix.png'
        plt.savefig(cm_plot_path, dpi=100, bbox_inches='tight')
        plt.close()

        # Plot and save training history
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Val Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        history_plot_path = Path(config['models_path']) / 'training_history.png'
        plt.savefig(history_plot_path, dpi=100, bbox_inches='tight')
        plt.close()

        # Log artifacts
        mlflow.log_artifact(str(metrics_file))
        mlflow.log_artifact(str(cm_plot_path))
        mlflow.log_artifact(str(history_plot_path))

        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE")
        print("="*70)
        print(f"Model saved: {model_path}")
        print(f"Metrics: {metrics_file}")
        print(f"Confusion Matrix: {cm_plot_path}")
        print(f"Training History: {history_plot_path}")


def main():
    parser = argparse.ArgumentParser(description='Train CNN model for cats vs dogs classification')
    parser.add_argument('--data-path', type=str, default='data/processed', help='Path to processed data')
    parser.add_argument('--models-path', type=str, default='models', help='Path to save models')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--experiment-name', type=str, default='cats_dogs_classification', help='MLflow experiment name')

    args = parser.parse_args()

    # Create models directory
    Path(args.models_path).mkdir(parents=True, exist_ok=True)

    config = {
        'data_path': args.data_path,
        'models_path': args.models_path,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'experiment_name': args.experiment_name
    }

    # Train model
    train_model(config)


if __name__ == "__main__":
    main()
