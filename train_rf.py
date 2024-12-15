import matplotlib.pyplot as plt
import csv
from PIL import Image
import numpy as np
from torchvision.transforms import transforms
import torchvision.transforms.functional as TF
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import pickle
from shapely.geometry import Polygon
import torch
import random
import os
import matplotlib.patches as patches

def set_all_seeds(seed=2):
    """Set seeds for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_all_seeds(2)

class LabelTransformer:
    def __init__(self):
        self.x_scale = None
        self.y_scale = None
        self.original_dims = None
        self.means = None
        self.stds = None

    def fit(self, labels, original_dims):
        """
        Calculate parameters needed for label transformation
        Args:
            labels: Original labels [x, y, width, height]
            original_dims: Original image dimensions (height, width)
        """
        self.original_dims = original_dims
        # Ensure conversion to numpy array
        labels = np.array(labels, dtype=np.float32)

        # Calculate scaling ratios
        self.x_scale = 32 / original_dims[1]  # width scale
        self.y_scale = 32 / original_dims[0]  # height scale

        # First perform coordinate transformation
        scaled_labels = self.scale_coordinates(labels)

        # Calculate normalization parameters
        self.means = np.mean(scaled_labels, axis=0)
        self.stds = np.std(scaled_labels, axis=0)

        return self

    def transform(self, labels):
        """
        Transform labels
        """
        # Ensure conversion to numpy array
        labels = np.array(labels, dtype=np.float32)

        # First perform coordinate transformation
        scaled = self.scale_coordinates(labels)

        # Then perform normalization
        normalized = (scaled - self.means) / self.stds

        return normalized

    def inverse_transform(self, normalized_labels):
        """
        Convert transformed labels back to original scale
        """
        # Ensure input is numpy array
        normalized_labels = np.array(normalized_labels, dtype=np.float32)

        # First perform denormalization
        scaled = normalized_labels * self.stds + self.means

        # Then perform inverse coordinate transformation
        original = np.copy(scaled)
        original[:, 0] = scaled[:, 0] / self.x_scale  # x
        original[:, 1] = scaled[:, 1] / self.y_scale  # y
        original[:, 2] = scaled[:, 2] / self.x_scale  # width
        original[:, 3] = scaled[:, 3] / self.y_scale  # height

        return original

    def scale_coordinates(self, labels):
        """
        Scale coordinates to 32x32 image size
        """
        # Ensure input is numpy array
        labels = np.array(labels, dtype=np.float32)

        scaled = np.copy(labels)
        scaled[:, 0] = labels[:, 0] * self.x_scale  # x
        scaled[:, 1] = labels[:, 1] * self.y_scale  # y
        scaled[:, 2] = labels[:, 2] * self.x_scale  # width
        scaled[:, 3] = labels[:, 3] * self.y_scale  # height
        return scaled

class DeterministicTransform:
    def __init__(self, seed=2):
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, img):
        # Create deterministic transformations
        # Horizontal flip
        if self.rng.random() < 0.5:
            img = TF.hflip(img)

        # Vertical flip
        if self.rng.random() < 0.5:
            img = TF.vflip(img)

        # Rotation
        angle = float(self.rng.uniform(-30, 30))
        img = TF.rotate(img, angle, fill=0)

        # Color adjustments
        brightness_factor = float(1.0 + self.rng.uniform(-0.2, 0.2))
        contrast_factor = float(1.0 + self.rng.uniform(-0.2, 0.2))
        saturation_factor = float(1.0 + self.rng.uniform(-0.2, 0.2))
        hue_factor = float(self.rng.uniform(-0.2, 0.2))

        img = TF.adjust_brightness(img, brightness_factor)
        img = TF.adjust_contrast(img, contrast_factor)
        img = TF.adjust_saturation(img, saturation_factor)
        img = TF.adjust_hue(img, hue_factor)

        # Convert to tensor and normalize
        img = TF.to_tensor(img)
        img = TF.normalize(img, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        return img

def readImageData(rootpath, deterministic_transform):
    '''Reads data with deterministic transformations'''
    images = []
    outputs = []
    filenames = []
    original_dims = None  # Store original image dimensions

    prefix = rootpath + '/'
    gtFile = open(prefix + 'mydata' + '.csv')
    gtReader = csv.reader(gtFile, delimiter=';')
    next(gtReader)

    # Process files in fixed order
    rows = sorted(list(gtReader), key=lambda x: x[0])

    for idx, row in enumerate(rows):
        img_seed = 2 + idx
        transform = DeterministicTransform(seed=img_seed)

        img = Image.open(prefix + row[0])
        if original_dims is None:
            original_dims = img.size[::-1]  # Save dimensions of first image (height, width)

        img = img.resize((32, 32), Image.BICUBIC)
        img = transform(img)
        img = np.array(img)

        images.append(img)
        outputs.append([float(row[1]), float(row[2]), float(row[3]), float(row[4])])
        filenames.append(row[0])

    gtFile.close()
    return images, outputs, filenames, original_dims

def visualize_predictions(image_paths, true_coords, pred_coords, num_samples=5):
    """
    Visualize prediction results and ground truth annotations
    Args:
        image_paths: List of image file paths
        true_coords: List of true coordinates [x, y, width, height]
        pred_coords: List of predicted coordinates [x, y, width, height]
        num_samples: Number of samples to visualize
    """
    # Randomly select samples
    indices = np.random.choice(len(image_paths), num_samples, replace=False)

    fig, axes = plt.subplots(1, num_samples, figsize=(20, 4))
    for idx, ax in enumerate(axes):
        # Read original image
        img = Image.open(image_paths[indices[idx]])
        ax.imshow(img)

        # Draw ground truth (green)
        true = true_coords[indices[idx]]
        true_ellipse = patches.Ellipse((true[0], true[1]), true[2], true[3],
                                       fill=False, color='g', label='Ground Truth')
        ax.add_patch(true_ellipse)

        # Draw prediction (red)
        pred = pred_coords[indices[idx]]
        pred_ellipse = patches.Ellipse((pred[0], pred[1]), pred[2], pred[3],
                                       fill=False, color='r', label='Prediction')
        ax.add_patch(pred_ellipse)

        ax.set_title(f'Sample {indices[idx]}')
        if idx == 0:  # Only add legend to first subplot
            ax.legend()

        # Turn off axes
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('prediction_visualization.png')
    plt.show()

def plot_error_distribution(true_coords, pred_coords):
    """
    Plot distribution of prediction errors
    """
    errors = np.abs(np.array(true_coords) - np.array(pred_coords))

    # Calculate errors for each dimension
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    labels = ['X coordinate', 'Y coordinate', 'Width', 'Height']

    for i, (ax, label) in enumerate(zip(axes.flat, labels)):
        ax.hist(errors[:, i], bins=30)
        ax.set_title(f'Error Distribution - {label}')
        ax.set_xlabel('Absolute Error')
        ax.set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig('error_distribution.png')
    plt.show()

def analyze_difficult_samples(true_coords, pred_coords, image_paths, threshold_percentile=90):
    """
    Analyze samples that are difficult to predict
    Args:
        true_coords: True coordinates
        pred_coords: Predicted coordinates
        image_paths: Image paths
        threshold_percentile: Error threshold percentile for defining "difficult" samples
    """
    # Calculate prediction error for each sample
    errors = np.sqrt(np.sum((true_coords - pred_coords) ** 2, axis=1))  # Euclidean distance

    # Calculate error threshold (e.g., top 10% high-error samples)
    error_threshold = np.percentile(errors, threshold_percentile)

    # Find indices of difficult samples
    difficult_indices = np.where(errors >= error_threshold)[0]

    # Analyze these samples
    print(f"\nAnalyzing difficult samples (error >= {error_threshold:.2f}):")
    print(f"Number of difficult samples: {len(difficult_indices)}")

    # Calculate average error for each dimension
    dim_errors = np.abs(true_coords - pred_coords)
    dim_names = ['X coordinate', 'Y coordinate', 'Width', 'Height']

    print("\nAverage errors for difficult samples:")
    for i, name in enumerate(dim_names):
        avg_error = np.mean(dim_errors[difficult_indices, i])
        print(f"{name}: {avg_error:.2f}")

    return difficult_indices, errors[difficult_indices]

def visualize_difficult_samples(image_paths, true_coords, pred_coords, difficult_indices, errors, num_samples=5):
    """
    Visualize samples that are difficult to predict
    """
    # Select samples with highest errors
    selected_indices = difficult_indices[np.argsort(errors)[-num_samples:]]

    fig, axes = plt.subplots(1, num_samples, figsize=(20, 4))
    for idx, ax in enumerate(axes):
        sample_idx = selected_indices[idx]

        # Read original image
        img = Image.open(image_paths[sample_idx])
        ax.imshow(img)

        # Draw ground truth (green)
        true = true_coords[sample_idx]
        true_ellipse = patches.Ellipse((true[0], true[1]), true[2], true[3],
                                       fill=False, color='g', label='Ground Truth')
        ax.add_patch(true_ellipse)

        # Draw prediction (red)
        pred = pred_coords[sample_idx]
        pred_ellipse = patches.Ellipse((pred[0], pred[1]), pred[2], pred[3],
                                       fill=False, color='r', label='Prediction')
        ax.add_patch(pred_ellipse)

        ax.set_title(f'Error: {errors[idx]:.2f}\nSample {sample_idx}')
        if idx == 0:
            ax.legend()

        ax.axis('off')

    plt.tight_layout()
    plt.savefig('difficult_samples_visualization.png')
    plt.show()

def analyze_image_characteristics(image_paths, difficult_indices):
    """
    Analyze image characteristics of difficult samples
    """
    characteristics = {
        'brightness': [],
        'contrast': [],
        'size': []
    }

    for idx in difficult_indices:
        img = Image.open(image_paths[idx])
        img_array = np.array(img)

        # Calculate brightness (average pixel value)
        brightness = np.mean(img_array)

        # Calculate contrast (pixel standard deviation)
        contrast = np.std(img_array)

        # Image size
        size = img_array.shape

        characteristics['brightness'].append(brightness)
        characteristics['contrast'].append(contrast)
        characteristics['size'].append(size)

    # Print analysis results
    print("\nCharacteristics of difficult samples:")
    print(f"Average brightness: {np.mean(characteristics['brightness']):.2f}")
    print(f"Average contrast: {np.mean(characteristics['contrast']):.2f}")

    return characteristics

# Calculate overlap area
def ellipse_overlap_area(pred, true):
    pred_ellipse = Polygon([(pred[0] - pred[2] / 2, pred[1] - pred[3] / 2),
                            (pred[0] + pred[2] / 2, pred[1] - pred[3] / 2),
                            (pred[0] + pred[2] / 2, pred[1] + pred[3] / 2),
                            (pred[0] - pred[2] / 2, pred[1] + pred[3] / 2)])
    true_ellipse = Polygon([(true[0] - true[2] / 2, true[1] - true[3] / 2),
                            (true[0] + true[2] / 2, true[1] - true[3] / 2),
                            (true[0] + true[2] / 2, true[1] + true[3] / 2),
                            (true[0] - true[2] / 2, true[1] + true[3] / 2)])
    overlap_area = pred_ellipse.intersection(true_ellipse).area
    true_area = true_ellipse.area
    return overlap_area / true_area

def main():
    # Read data and apply deterministic transformations
    trainImages, trainOutputs, trainFilenames, original_dims = readImageData(
        'Wound/Training',
        deterministic_transform=True
    )
    print('number of historical data=', len(trainOutputs))

    # Prepare model inputs
    X = np.array([img.flatten() for img in trainImages])
    Y = np.array(trainOutputs, dtype=np.float32)  # Ensure conversion to numpy array

    # Transform labels
    label_transformer = LabelTransformer()
    label_transformer.fit(trainOutputs, original_dims)
    Y_transformed = label_transformer.transform(trainOutputs)

    # Save label transformer for later use
    pickle.dump(label_transformer, open('label_transformer.sav', 'wb'))

    # Train model
    reg = RandomForestRegressor(n_estimators=100,
                                random_state=2,
                                n_jobs=-1)
    reg.fit(X, Y_transformed)

    # Predict and convert back to original scale
    Ypred_transformed = reg.predict(X)
    Ypred = label_transformer.inverse_transform(Ypred_transformed)

    # Evaluate results
    MSE = mean_squared_error(Y, Ypred)
    print('\nTraining MSE=', MSE)

    overlap_ratios = [ellipse_overlap_area(pred, true) for pred, true in zip(Ypred, Y)]
    avg_overlap_ratio = np.mean(overlap_ratios)
    print(f"Average Overlap Ratio: {avg_overlap_ratio}")

    # Visualize prediction results
    image_paths = [f'Wound/Training/{filename}'
                   for filename in trainFilenames]
    visualize_predictions(image_paths, Y, Ypred)

    # Plot error distribution
    plot_error_distribution(Y, Ypred)

    # Analyze difficult samples
    difficult_indices, difficult_errors = analyze_difficult_samples(Y, Ypred, image_paths)

    # Visualize difficult samples
    visualize_difficult_samples(image_paths, Y, Ypred, difficult_indices, difficult_errors)

    # Analyze image characteristics of difficult samples
    characteristics = analyze_image_characteristics(image_paths, difficult_indices)

    # Save detailed information about difficult samples
    difficult_samples_info = {
        'indices': difficult_indices,
        'errors': difficult_errors,
        'characteristics': characteristics
    }
    pickle.dump(difficult_samples_info, open('difficult_samples_analysis.sav', 'wb'))

    # Save model
    pickle.dump(reg, open('model_rf.sav', 'wb'))


if __name__ == "__main__":
    main()