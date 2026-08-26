"""
Inference utilities for model prediction
"""

import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class CatsDogsClassifier:
    """Model wrapper for inference"""

    def __init__(self, model_path, image_size=224):
        """
        Initialize classifier with trained model

        Args:
            model_path: Path to trained model (.h5 file)
            image_size: Input image size (default 224x224)
        """
        self.model_path = Path(model_path)
        self.image_size = image_size
        self.model = None
        self.class_names = ['cat', 'dog']

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        self.load_model()

    def load_model(self):
        """Load pre-trained model"""
        try:
            logger.info(f"Loading model from {self.model_path}")
            self.model = keras.models.load_model(str(self.model_path))
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def preprocess_image(self, image_path):
        """
        Load and preprocess image

        Args:
            image_path: Path to image file or PIL Image

        Returns:
            Preprocessed image array ready for prediction
        """
        try:
            # Load image
            if isinstance(image_path, str) or isinstance(image_path, Path):
                image = Image.open(image_path)
            else:
                image = image_path

            # Convert to RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Resize
            image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)

            # Convert to array and normalize
            img_array = np.array(image, dtype=np.float32) / 255.0

            return img_array
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise

    def predict(self, image_path, return_probs=True):
        """
        Predict class for image

        Args:
            image_path: Path to image or PIL Image
            return_probs: Whether to return probabilities

        Returns:
            Dictionary with prediction results
        """
        try:
            # Preprocess
            img_array = self.preprocess_image(image_path)

            # Add batch dimension
            img_batch = np.expand_dims(img_array, axis=0)

            # Predict
            predictions = self.model.predict(img_batch, verbose=0)
            probabilities = predictions[0]

            # Get class
            class_idx = np.argmax(probabilities)
            class_name = self.class_names[class_idx]
            confidence = float(probabilities[class_idx])

            result = {
                'predicted_class': class_name,
                'confidence': confidence,
                'class_index': int(class_idx)
            }

            if return_probs:
                result['probabilities'] = {
                    'cat': float(probabilities[0]),
                    'dog': float(probabilities[1])
                }

            return result
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise

    def batch_predict(self, image_paths):
        """
        Predict on batch of images

        Args:
            image_paths: List of image paths

        Returns:
            List of prediction results
        """
        results = []
        for img_path in image_paths:
            try:
                result = self.predict(img_path)
                results.append({
                    'image': str(img_path),
                    'prediction': result
                })
            except Exception as e:
                logger.error(f"Error predicting for {img_path}: {e}")
                results.append({
                    'image': str(img_path),
                    'error': str(e)
                })

        return results


def get_model_info(model_path):
    """Get model information"""
    try:
        model = keras.models.load_model(model_path)
        return {
            'parameters': int(model.count_params()),
            'layers': len(model.layers),
            'input_shape': model.input_shape,
            'output_shape': model.output_shape
        }
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return None
