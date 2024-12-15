import matplotlib.pyplot as plt
import csv
from PIL import Image, ImageFilter
import numpy as np
from torchvision.transforms import transforms
from torchvision.transforms import functional as F
from sklearn.metrics import mean_squared_error
import pickle
from shapely.geometry import Polygon
import xgboost as xgb
import os
import random
import torch
import matplotlib.patches as patches


# Set all relevant random seeds for reproducibility
def set_all_seeds(seed=7):
    """Set seeds for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)  # Set Python's hash seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Set random seeds at the beginning of the program
set_all_seeds(7)


def readImageData(rootpath, transform=None, augment_times=1):
    '''Reads data
    Arguments: path to the image, for example './Training'
    Returns:   list of images, list of corresponding outputs'''
    images = []  # images
    outputs = []  # corresponding outputs (x, y, x_width, y_width)
    filenames = []  # store filenames

    prefix = rootpath + '/'
    gtFile = open(prefix + 'mydata' + '.csv')  # annotations file
    gtReader = csv.reader(gtFile, delimiter=';')  # csv parser for annotations file
    next(gtReader)
    # loop over all images in current annotations file
    for row in gtReader:
        img = Image.open(prefix + row[0])  # the 1th column is the filename
        img = img.resize((32, 32), Image.BICUBIC)
        # Data augmentation
        for _ in range(augment_times):
            if transform:
                augmented_img, label = transform(img, [float(row[1]), float(row[2]), float(row[3]), float(row[4])])
            else:
                augmented_img = np.array(img)
                label = [float(row[1]), float(row[2]), float(row[3]), float(row[4])]
            images.append(augmented_img)
            outputs.append(label)  # the 2nd to 5th columns are the labels
            filenames.append(row[0])  # save filename
    gtFile.close()
    return images, outputs, filenames


# Custom data augmentation transformation, including corresponding label transformations
class CustomTransform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.6396, 0.6095, 0.6041), (0.2238, 0.2355, 0.2415))
        ])

    def __call__(self, img, label):
        if np.random.rand() > 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=np.random.uniform(0.1, 2.0)))

        # Initial image transformation
        img = self.transform(img)

        # Random horizontal flip
        if np.random.rand() > 0.5:
            img = F.hflip(img)
            label[0] = 3264 - label[0]

        # Random vertical flip
        if np.random.rand() > 0.5:
            img = F.vflip(img)
            label[1] = 2448 - label[1]

        # Random rotation
        if np.random.rand() > 0.5:
            angle = np.random.uniform(-30, 30)
            img = F.rotate(img, angle)
            radians = np.deg2rad(angle)
            x, y = label[0] - 1632, label[1] - 1224  # Move center point to origin
            new_x = x * np.cos(radians) - y * np.sin(radians)
            new_y = x * np.sin(radians) + y * np.cos(radians)
            label[0], label[1] = new_x + 1632, new_y + 1224  # Move center point back

        # Random color jitter
        if np.random.rand() > 0.5:
            color_jitter = transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.2, hue=0.2)
            img = color_jitter(img)

        return img, label


transform = CustomTransform()

# Use only augmented data
trainImages, trainOutputs, trainFilenames = readImageData('Wound\\Training', transform=transform, augment_times=5)

# print number of historical images
print('number of historical data=', len(trainOutputs))

# design the input and output for model
X = []
Y = []
for i in range(0, len(trainOutputs)):
    # input X just the flattern image, you can design other features to represent a image
    X.append(trainImages[i].flatten())
    Y.append(trainOutputs[i])
X = np.array(X)
Y = np.array(Y)


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


# train an XGBoost model
dtrain = xgb.DMatrix(X, label=Y)
params = {
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'eta': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eval_metric': 'rmse',
    'seed': 7  # Set random seed
}
num_boost_round = 100
model = xgb.train(params, dtrain, num_boost_round)

# Prediction
dtrain = xgb.DMatrix(X)
Ypred = model.predict(dtrain)

# check the accuracy
MSE = mean_squared_error(Y, Ypred)
print('Training MSE=', MSE)

# Calculate overlap ratio
overlap_ratios = [ellipse_overlap_area(pred, true) for pred, true in zip(Ypred, Y)]
avg_overlap_ratio = np.mean(overlap_ratios)
print(f"Average Overlap Ratio: {avg_overlap_ratio}")

# save model
model.save_model('model_xgboost.json')


# Visualize prediction results
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

        # Draw ground truth annotation (green)
        true = true_coords[indices[idx]]
        true_ellipse = patches.Ellipse((true[0], true[1]), true[2], true[3],
                                       fill=False, color='g', label='Ground Truth')
        ax.add_patch(true_ellipse)

        # Draw prediction result (red)
        pred = pred_coords[indices[idx]]
        pred_ellipse = patches.Ellipse((pred[0], pred[1]), pred[2], pred[3],
                                       fill=False, color='r', label='Prediction')
        ax.add_patch(pred_ellipse)

        ax.set_title(f'Sample {indices[idx]}')
        if idx == 0:  # Only add legend to the first subplot
            ax.legend()

        # Turn off axes
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('prediction_visualization.png')
    plt.show()


# Plot error distribution
def plot_error_distribution(true_coords, pred_coords):
    """
    Plot prediction error distribution
    """
    errors = np.abs(np.array(true_coords) - np.array(pred_coords))

    # Calculate error for each dimension
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


# Analyze difficult samples
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


# Visualize difficult samples
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

        # Draw ground truth annotation (green)
        true = true_coords[sample_idx]
        true_ellipse = patches.Ellipse((true[0], true[1]), true[2], true[3],
                                       fill=False, color='g', label='Ground Truth')
        ax.add_patch(true_ellipse)

        # Draw prediction result (red)
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


# Analyze image characteristics of difficult samples
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

        # Calculate brightness (mean pixel value)
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


# Visualize prediction results
image_paths = [f'Wound/Training/{filename}' for filename in trainFilenames]
visualize_predictions(image_paths, Y, Ypred)

# Plot error distribution
plot_error_distribution(Y, Ypred)

# Analyze difficult samples
difficult_indices, difficult_errors = analyze_difficult_samples(Y, Ypred, image_paths)

# Visualize difficult samples
visualize_difficult_samples(image_paths, Y, Ypred, difficult_indices, difficult_errors)

# Analyze image characteristics of difficult samples
characteristics = analyze_image_characteristics(image_paths, difficult_indices)

# Save detailed information of difficult samples
difficult_samples_info = {
    'indices': difficult_indices,
    'errors': difficult_errors,
    'characteristics': characteristics
}
pickle.dump(difficult_samples_info, open('difficult_samples_analysis.sav', 'wb'))