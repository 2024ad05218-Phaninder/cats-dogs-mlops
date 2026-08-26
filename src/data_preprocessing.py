"""
Data Preprocessing Script for Cats vs Dogs Classification (Fixed - Handles File Limits)
Properly closes files to avoid "Too many open files" error
"""

import os
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from PIL import Image
import json
from datetime import datetime
import argparse
from tqdm import tqdm

class DataPreprocessor:
    def __init__(self, raw_data_path, processed_data_path, image_size=224):
        self.raw_data_path = Path(raw_data_path)
        self.processed_data_path = Path(processed_data_path)
        self.image_size = image_size
        self.stats = {}

    def load_image_paths(self):
        """Load image paths without keeping files open"""
        print("Collecting image paths...")
        image_paths = []
        labels = []

        for label, category in enumerate(['Cat', 'Dog']):
            category_path = self.raw_data_path / category

            if not category_path.exists():
                category_path = self.raw_data_path / category.lower()

            if not category_path.exists():
                raise FileNotFoundError(f"Category {category} not found at {self.raw_data_path}")

            image_files = list(category_path.glob('*.jpg')) + list(category_path.glob('*.png'))
            print(f"Found {len(image_files)} {category} image paths")

            for img_path in image_files:
                image_paths.append(img_path)
                labels.append(label)

        return image_paths, np.array(labels)

    def load_and_process_image(self, img_path):
        """Load, process, and close image - keeps memory clean"""
        try:
            # Load image
            img = Image.open(img_path)

            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Resize to target size
            img = img.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)

            # Return as array to close file handle
            return np.array(img)
        except Exception as e:
            return None

    def augment_image(self, img_array, augment=True):
        """Apply data augmentation to numpy array"""
        if not augment or img_array is None:
            return img_array

        try:
            # Random horizontal flip (50%)
            if np.random.rand() > 0.5:
                img_array = np.fliplr(img_array)

            # Random brightness adjustment
            brightness = np.random.uniform(0.85, 1.15)
            img_array = np.clip(img_array * brightness, 0, 255).astype(np.uint8)

            # Random contrast adjustment
            if np.random.rand() > 0.5:
                contrast = np.random.uniform(0.85, 1.15)
                mean = np.mean(img_array)
                img_array = np.clip((img_array - mean) * contrast + mean, 0, 255).astype(np.uint8)

            return img_array
        except Exception as e:
            return img_array

    def save_image(self, img_array, save_path):
        """Save numpy array as image"""
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            img = Image.fromarray(img_array.astype(np.uint8))
            img.save(save_path, quality=95)
            return True
        except Exception as e:
            return False

    def process_and_split(self, train_size=0.8, val_size=0.1, test_size=0.1, seed=42):
        """Process images and split into train/val/test"""
        print("\n" + "="*70)
        print("DATA PREPROCESSING PIPELINE")
        print("="*70)

        # Validate split sizes
        assert train_size + val_size + test_size == 1.0, "Split sizes must sum to 1.0"

        # Load image paths (not files)
        image_paths, labels = self.load_image_paths()
        print(f"\n📊 Total image paths collected: {len(image_paths)}")
        print(f"📊 Class distribution: Cats={np.sum(labels==0)}, Dogs={np.sum(labels==1)}")

        # Split indices
        indices = np.arange(len(image_paths))
        train_idx, temp_idx = train_test_split(
            indices,
            test_size=(val_size + test_size),
            random_state=seed,
            stratify=labels
        )

        val_size_adjusted = val_size / (val_size + test_size)
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=1 - val_size_adjusted,
            random_state=seed,
            stratify=labels[temp_idx]
        )

        splits = {
            'train': train_idx,
            'validation': val_idx,
            'test': test_idx
        }

        # Process and save images
        split_stats = {}
        for split_name, indices in splits.items():
            print(f"\n🔄 Processing {split_name} set ({len(indices)} images)...")
            split_stats[split_name] = {'cats': 0, 'dogs': 0, 'count': 0, 'failed': 0}

            for idx in tqdm(indices, desc=f"  {split_name}", ncols=80):
                img_path = image_paths[idx]
                label = labels[idx]
                category = 'cats' if label == 0 else 'dogs'

                # Load and process image (file handle is closed after)
                img_array = self.load_and_process_image(img_path)
                if img_array is None:
                    split_stats[split_name]['failed'] += 1
                    continue

                # Apply augmentation for training set
                if split_name == 'train':
                    img_array = self.augment_image(img_array, augment=True)
                else:
                    img_array = self.augment_image(img_array, augment=False)

                # Save
                save_dir = self.processed_data_path / split_name / category
                save_path = save_dir / img_path.name

                if self.save_image(img_array, save_path):
                    split_stats[split_name][category] += 1
                    split_stats[split_name]['count'] += 1
                else:
                    split_stats[split_name]['failed'] += 1

            print(f"✅ {split_name}: {split_stats[split_name]['count']} images saved")
            if split_stats[split_name]['cats'] > 0:
                print(f"   - Cats: {split_stats[split_name]['cats']}")
            if split_stats[split_name]['dogs'] > 0:
                print(f"   - Dogs: {split_stats[split_name]['dogs']}")
            if split_stats[split_name]['failed'] > 0:
                print(f"   ⚠️  Failed: {split_stats[split_name]['failed']}")

        self.stats = split_stats
        self._save_stats(train_size, val_size, test_size)
        return split_stats

    def _save_stats(self, train_size, val_size, test_size):
        """Save preprocessing statistics"""
        stats_file = self.processed_data_path / 'data_stats.json'
        stats_file.parent.mkdir(parents=True, exist_ok=True)

        total = sum(s['count'] for s in self.stats.values())

        stats_to_save = {
            'timestamp': datetime.now().isoformat(),
            'image_size': self.image_size,
            'splits': {
                'train': {'ratio': train_size, **self.stats['train']},
                'validation': {'ratio': val_size, **self.stats['validation']},
                'test': {'ratio': test_size, **self.stats['test']}
            },
            'total_images': total,
            'class_distribution': {
                'cats': sum(s['cats'] for s in self.stats.values()),
                'dogs': sum(s['dogs'] for s in self.stats.values())
            }
        }

        with open(stats_file, 'w') as f:
            json.dump(stats_to_save, f, indent=2)

        print(f"\n📝 Statistics saved to {stats_file}")

def main():
    parser = argparse.ArgumentParser(description='Preprocess Cats vs Dogs dataset')
    parser.add_argument('--raw-path', type=str, default='data/raw', help='Path to raw data')
    parser.add_argument('--processed-path', type=str, default='data/processed', help='Path to save processed data')
    parser.add_argument('--image-size', type=int, default=224, help='Target image size')
    parser.add_argument('--train-split', type=float, default=0.8, help='Training set ratio')
    parser.add_argument('--val-split', type=float, default=0.1, help='Validation set ratio')
    parser.add_argument('--test-split', type=float, default=0.1, help='Test set ratio')

    args = parser.parse_args()

    # Initialize preprocessor
    preprocessor = DataPreprocessor(
        args.raw_path,
        args.processed_path,
        image_size=args.image_size
    )

    # Process and split
    stats = preprocessor.process_and_split(
        train_size=args.train_split,
        val_size=args.val_split,
        test_size=args.test_split
    )

    print("\n" + "="*70)
    print("✅ DATA PREPROCESSING COMPLETE")
    print("="*70)
    print(f"\n📁 Processed data saved to: {args.processed_path}")
    print(f"📊 Train: {stats['train']['count']} images")
    print(f"📊 Validation: {stats['validation']['count']} images")
    print(f"📊 Test: {stats['test']['count']} images")
    print(f"📊 Total: {sum(s['count'] for s in stats.values())} images")
    print("\nNext: Run 'dvc add data/processed' to track with DVC")

if __name__ == "__main__":
    main()
