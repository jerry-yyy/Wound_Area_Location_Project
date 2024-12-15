import os
import csv
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import transforms
from sklearn.metrics import mean_squared_error
import pickle
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# Define LabelTransformer class before loading the pickled transformer
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


# Test data folder path
test_image_folder = 'Wound/Test'


# Read image data and labels
def readImageData(rootpath, transform):
    '''Reads data
    Arguments: path to the image, for example './Test'
    Returns:   list of images, list of corresponding outputs, list of filenames'''
    images = []  # images
    outputs = []  # corresponding outputs (x, y, x_width, y_width)
    filenames = []  # store filenames
    original_dims = None  # Store original image dimensions

    prefix = rootpath + '/'
    gtFile = open(prefix + 'mydata' + '.csv')  # annotations file
    gtReader = csv.reader(gtFile, delimiter=';')  # csv parser for annotations file
    next(gtReader)
    # loop over all images in current annotations file
    for row in gtReader:
        img_path = prefix + row[0]
        if not os.path.exists(img_path):
            print(f"File not found: {img_path}")
            continue

        img = Image.open(img_path)  # the 1th column is the filename
        if original_dims is None:
            original_dims = img.size[::-1]  # Save dimensions of first image (height, width)

        img = img.resize((32, 32), Image.BICUBIC)
        if transform:
            img = transform(img)
        img = np.array(img)
        images.append(img)
        outputs.append(
            [float(row[1]), float(row[2]), float(row[3]), float(row[4])])  # the 2nd to 5th columns are the labels
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
        if not os.path.exists(image_paths[indices[idx]]):
            print(f"File not found: {image_paths[indices[idx]]}")
            continue

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

        ax.axis('off')

    plt.tight_layout()
    plt.savefig('test_prediction_visualization_rf.png')
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
    plt.savefig('test_error_distribution_rf.png')
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
    num_samples = min(num_samples, len(difficult_indices))
    selected_indices = difficult_indices[np.argsort(errors)[-num_samples:]]

    fig, axes = plt.subplots(1, num_samples, figsize=(20, 4))
    for idx, ax in enumerate(axes):
        sample_idx = selected_indices[idx]

        if not os.path.exists(image_paths[sample_idx]):
            print(f"File not found: {image_paths[sample_idx]}")
            continue

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
    plt.savefig('test_difficult_samples_visualization_rf.png')
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
        if not os.path.exists(image_paths[idx]):
            print(f"File not found: {image_paths[idx]}")
            continue

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


# Data preprocessing transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load test data
X_test, y_true, testFilenames, original_dims = readImageData(test_image_folder, transform=transform)
X_test = np.array(X_test)
y_true = np.array(y_true, dtype=np.float32)  # Ensure conversion to numpy array

# Flatten image data
X_test_flatten = X_test.reshape(X_test.shape[0], -1)

# Load trained random forest model and label transformer
with open('model_rf.sav', 'rb') as model_file:
    rf_model = pickle.load(model_file)

with open('label_transformer.sav', 'rb') as transformer_file:
    label_transformer = pickle.load(transformer_file)


# Calculate overlap area between two ellipses
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


# Predictions
# Use model to make predictions (model expects transformed label format)
predictions_transformed = rf_model.predict(X_test_flatten)
# Convert predictions back to original scale
predictions_rf = label_transformer.inverse_transform(predictions_transformed)

# Calculate and output performance metrics
mse_rf = mean_squared_error(y_true, predictions_rf)
overlap_ratios = [ellipse_overlap_area(pred, true) for pred, true in zip(predictions_rf, y_true)]
avg_overlap_ratio = np.mean(overlap_ratios)

print(f"Random Forest MSE: {mse_rf}")
print(f"Average Overlap Ratio: {avg_overlap_ratio}")

# Visualize prediction results
image_paths = [f'{test_image_folder}/{filename}' for filename in testFilenames]
visualize_predictions(image_paths, y_true, predictions_rf)

# Plot error distribution
plot_error_distribution(y_true, predictions_rf)

# Analyze difficult samples
difficult_indices, difficult_errors = analyze_difficult_samples(y_true, predictions_rf, image_paths)

# Visualize difficult samples
visualize_difficult_samples(image_paths, y_true, predictions_rf, difficult_indices, difficult_errors)

# Analyze image characteristics of difficult samples
characteristics = analyze_image_characteristics(image_paths, difficult_indices)

# Save detailed information about difficult samples
difficult_samples_info = {
    'indices': difficult_indices,
    'errors': difficult_errors,
    'characteristics': characteristics
}
pickle.dump(difficult_samples_info, open('test_difficult_samples_analysis_rf.sav', 'wb'))