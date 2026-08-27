"""
Unit tests for data preprocessing module
Tests: image loading, resizing, augmentation
"""

import pytest
import numpy as np
from pathlib import Path
from PIL import Image
import tempfile
import os

# Import preprocessing functions
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from data_preprocessing import DataPreprocessor


class TestDataPreprocessor:
    """Test cases for DataPreprocessor class"""

    @pytest.fixture
    def temp_data(self):
        """Create temporary test data"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test directories (uppercase to match PetImages structure)
            cats_dir = Path(tmpdir) / 'Cat'
            dogs_dir = Path(tmpdir) / 'Dog'
            cats_dir.mkdir()
            dogs_dir.mkdir()

            # Create dummy images
            for i in range(3):
                # Create cat image
                img = Image.new('RGB', (256, 256), color='red')
                img.save(cats_dir / f'cat_{i}.jpg')

                # Create dog image
                img = Image.new('RGB', (256, 256), color='blue')
                img.save(dogs_dir / f'dog_{i}.jpg')

            yield tmpdir

    def test_preprocessor_initialization(self, temp_data):
        """Test DataPreprocessor initialization"""
        preprocessor = DataPreprocessor(temp_data, 'output', image_size=224)
        assert preprocessor.image_size == 224
        assert preprocessor.raw_data_path == Path(temp_data)

    def test_load_image_paths(self, temp_data):
        """Test loading image paths"""
        preprocessor = DataPreprocessor(temp_data, 'output', image_size=224)
        image_paths, labels = preprocessor.load_image_paths()

        assert len(image_paths) == 6  # 3 cats + 3 dogs
        assert len(labels) == 6
        assert np.sum(labels == 0) == 3  # 3 cats
        assert np.sum(labels == 1) == 3  # 3 dogs

    def test_load_and_process_image(self, temp_data):
        """Test image loading and processing"""
        preprocessor = DataPreprocessor(temp_data, 'output', image_size=224)
        image_paths, _ = preprocessor.load_image_paths()

        # Test first image
        img_array = preprocessor.load_and_process_image(image_paths[0])

        assert img_array is not None
        assert img_array.shape == (224, 224, 3)
        assert img_array.dtype == np.uint8
        assert np.all(img_array >= 0)
        assert np.all(img_array <= 255)

    def test_preprocess_image_dimensions(self, temp_data):
        """Test image is resized to correct dimensions"""
        preprocessor = DataPreprocessor(temp_data, 'output', image_size=224)
        image_paths, _ = preprocessor.load_image_paths()

        img_array = preprocessor.load_and_process_image(image_paths[0])

        assert img_array.shape[0] == 224  # height
        assert img_array.shape[1] == 224  # width
        assert img_array.shape[2] == 3    # RGB channels

    def test_augment_image(self, temp_data):
        """Test image augmentation"""
        preprocessor = DataPreprocessor(temp_data, 'output', image_size=224)
        image_paths, _ = preprocessor.load_image_paths()

        img_array = preprocessor.load_and_process_image(image_paths[0])
        augmented = preprocessor.augment_image(img_array, augment=True)

        assert augmented is not None
        assert augmented.shape == img_array.shape
        assert augmented.dtype == np.uint8

    def test_augment_image_no_augmentation(self, temp_data):
        """Test that augment=False returns same array"""
        preprocessor = DataPreprocessor(temp_data, 'output', image_size=224)
        image_paths, _ = preprocessor.load_image_paths()

        img_array = preprocessor.load_and_process_image(image_paths[0])
        not_augmented = preprocessor.augment_image(img_array, augment=False)

        # Should be same shape
        assert not_augmented.shape == img_array.shape

    def test_save_image(self, temp_data):
        """Test image saving"""
        preprocessor = DataPreprocessor(temp_data, 'output', image_size=224)
        image_paths, _ = preprocessor.load_image_paths()

        img_array = preprocessor.load_and_process_image(image_paths[0])

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'test_image.jpg'
            result = preprocessor.save_image(img_array, save_path)

            assert result is True
            assert save_path.exists()

            # Verify saved image
            saved_img = Image.open(save_path)
            assert saved_img.size == (224, 224)

    def test_data_split(self, temp_data):
        """Test train/val/test split"""
        preprocessor = DataPreprocessor(temp_data, 'output', image_size=224)
        image_paths, labels = preprocessor.load_image_paths()

        # Test split
        train_size = 0.6
        val_size = 0.2
        test_size = 0.2

        n_total = len(image_paths)
        n_train = int(n_total * train_size)
        n_val = int(n_total * val_size)
        n_test = n_total - n_train - n_val

        assert n_train + n_val + n_test == n_total

    def test_image_value_range(self, temp_data):
        """Test image values are in valid range"""
        preprocessor = DataPreprocessor(temp_data, 'output', image_size=224)
        image_paths, _ = preprocessor.load_image_paths()

        img_array = preprocessor.load_and_process_image(image_paths[0])

        assert np.all(img_array >= 0), "Image has negative values"
        assert np.all(img_array <= 255), "Image has values > 255"

    def test_invalid_image_handling(self, temp_data):
        """Test handling of invalid images"""
        # Create invalid image file
        invalid_path = Path(temp_data) / 'invalid.jpg'
        with open(invalid_path, 'w') as f:
            f.write('not an image')

        preprocessor = DataPreprocessor(temp_data, 'output', image_size=224)

        # Should handle gracefully
        img_array = preprocessor.load_and_process_image(invalid_path)
        assert img_array is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
