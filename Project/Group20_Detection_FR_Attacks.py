import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from PIL import Image
import zipfile
import torchattacks
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, RandomHorizontalFlip, RandomRotation, ColorJitter, GaussianBlur, RandomAffine, ToTensor, Normalize, RandomPerspective, RandomErasing
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
from tqdm import tqdm
import json
import pickle
import sklearn
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import cv2
from scipy.ndimage import gaussian_filter, median_filter


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)

# Load data

transform_train = Compose([
    Resize((224, 224)),  # Resize images to a fixed size
    RandomHorizontalFlip(p=0.5),  # Flip images horizontally with a 50% chance
    RandomRotation(15),  # Rotate images randomly up to ±15 degrees
    ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),  # Adjust brightness, contrast, etc.
    RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Apply affine transformations
    RandomPerspective(distortion_scale=0.5, p=0.5),  # Simulate perspective distortion
    GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),  # Apply Gaussian blur
    RandomErasing(p=0.5),  # Randomly erase parts of the image
    ToTensor(),  # Convert image to PyTorch tensor
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize to ImageNet standards
])

transform_test = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data_dir = './data'
dataset = datasets.ImageFolder(root=data_dir)
num_classes = len(dataset.classes)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_dataset.dataset.transform = transform_train
test_dataset.dataset.transform = transform_test

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Model setup
model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

# Modify the classifier to suit the new task
model.classifier[5] = nn.Dropout(p=0.5)
model.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes, bias=True)

# Unfreeze the last 8 layers in the feature extractor
for param in model.features[:-8].parameters():  # Freeze earlier layers
    param.requires_grad = False
for param in model.features[-8:].parameters():  # Unfreeze later layers
    param.requires_grad = True

# Move the model to the device
model = model.to(device)

state_dict = torch.load("./vgg16_model.pth")
model.load_state_dict(state_dict)

print("Model weights successfully loaded!")

print(f"Test dataset size = {len(test_dataset)}")

################################## ADVERSARIAL ATTACK ######################################

def fgsm_attack(model, data_loader, eps):

    attack = torchattacks.FGSM(model, eps=eps)

    successful_adv_images_fgsm = []

    total_images = 0
    misclassified_images = 0

    for images, labels in tqdm(data_loader, desc="Generating Adversarial Images", unit="batch"):
        images = images.to(device)
        labels = labels.to(device)

        original_logits = model(images)
        _, original_preds = torch.max(original_logits, 1)

        correctly_classified_indices = (original_preds == labels).nonzero(as_tuple=True)[0]

        if len(correctly_classified_indices) == 0:
            continue

        # Select only correctly classified images and labels
        images = images[correctly_classified_indices]
        labels = labels[correctly_classified_indices]

        # Generate adversarial images
        adv_images = attack(images, labels)

        # Get model predictions for adversarial images
        adv_logits = model(adv_images)
        _, adv_preds = torch.max(adv_logits, 1)

        # Update counters
        total_images += images.size(0)
        misclassified_images += (adv_preds != labels).sum().item()

        # Store successful adversarial examples
        for i in range(images.size(0)):
            if adv_preds[i] != labels[i]:  # Misclassification due to the adversarial attack
                successful_adv_images_fgsm.append({
                    'original_image': images[i].cpu().numpy(),
                    'adv_image': adv_images[i].cpu().numpy(),
                    'original_label': labels[i].cpu().item(),
                    'adv_label': adv_preds[i].cpu().item(),
                })

    attack_success_rate = misclassified_images / total_images * 100
    print(f"Attack success rate: {attack_success_rate:.2f}%")

    return successful_adv_images_fgsm

test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
successful_adv_images_fgsm = fgsm_attack(model, test_loader, 0.01)

# Directory where the images and metadata are saved
output_dir = 'output_images'
metadata_file = os.path.join(output_dir, 'metadata.json')

with open(metadata_file, 'r') as f:
    metadata_list = json.load(f)

# Load and display the original and adversarial images (just as an example)
for metadata in metadata_list[:2]:
    original_image_path = metadata['original_image_path']
    adv_image_path = metadata['adv_image_path']

    original_img = np.load(original_image_path)
    adv_img = np.load(adv_image_path)

    original_img = np.transpose(original_img, (1, 2, 0))
    adv_img = np.transpose(adv_img, (1, 2, 0))

    plt.figure(figsize=(10, 5))
    
    # Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.title(f"Original Label: {metadata['original_label']}")
    plt.axis('off')

    # Adversarial Image
    plt.subplot(1, 2, 2)
    plt.imshow(adv_img)
    plt.title(f"Adversarial Label: {metadata['adv_label']}")
    plt.axis('off')

    plt.show()

######################################### DETECTION MODEL #########################################

model.eval()
model.to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

def get_layer_name_from_object(model, target_layer):
    for name, layer in model.named_modules():
        if layer is target_layer:
            return name
    return None

def compute_mean_representations(model, data_loader, device):
    activations = {}

    def hook_fn(module, input, output):
        activations[module] = output.detach()

    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            layer.register_forward_hook(hook_fn)

    mean_representations = {}

    mean_representations[str(layer)] = torch.zeros_like(next(layer.parameters()))

    with torch.no_grad():
        for images, _ in tqdm(data_loader, desc="Computing Mean Representations"):
            images = images.to(device)

            activations.clear()
            model(images)

            # print(activations)
            for layer, activation in activations.items():
                # print(f"Layer: {layer} | Activation: {activation}")
                name = get_layer_name_from_object(model, layer)
                if name in mean_representations:
                    # print(f"Mean Representation shape: {mean_representations[name].shape}, Activation shape: {activation.shape}")
                    # Sum activations over the batch dimension (dim=0)
                    # print(f"Layer: {layer} | Activation: {activation}")
                    a = activation.sum(dim=0)
                    mean_representations[name] += a
                else:
                    a = activation.sum(dim=0)
                    mean_representations[name] = a
                    # print(f"Mean Representation shape: {mean_representations[name].shape}, Activation shape: {activation.shape}")
                    # print(f"Layer {layer} not found in mean_representations.")
                    # return

    for layer, sum_activation in mean_representations.items():
        mean_representations[layer] = sum_activation / len(data_loader.dataset)

    return mean_representations

def save_mean_representations(mean_representations, file_path):
    mean_representations_cpu = {k: v.cpu() for k, v in mean_representations.items()}  # Move tensors to CPU
    with open(file_path, "wb") as f:
        pickle.dump(mean_representations_cpu, f)
    print(f"Mean representations saved successfully to {file_path}.")

def load_mean_representations(file_path, device="cpu"):
    with open(file_path, "rb") as f:
        mean_representations = pickle.load(f)
    mean_representations = {k: v.to(device) for k, v in mean_representations.items()}  # Move tensors to the specified device
    print(f"Mean representations loaded successfully from {file_path}.")
    return mean_representations

# Compute the mean representations
mean_representations = compute_mean_representations(model, train_loader, device)


for layer, mean_rep in mean_representations.items():
    print(f"Layer: {layer} | Mean Representation Shape: {mean_rep.shape}")

print("Mean representations computed successfully!")

save_mean_representations(mean_representations, "mean_representations.pkl")

mean_representations = load_mean_representations("mean_representations.pkl", device)

# print(mean_representations)

print("Successfully loaded mean representations!")

def register_hooks_for_layers(model):
    activations = {}

    def hook_fn(module, input, output):
        activations[module] = output.detach()

    hooks = []
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Conv2d):
            hook = layer.register_forward_hook(hook_fn)
            hooks.append(hook)

    return hooks, activations

def compute_canberra_distance_layer(activation, mean_representation, layer_name):
    activation = activation.mean(dim=0, keepdim=False)

    if activation.shape != mean_representation.shape:
        raise ValueError(f"Shape mismatch for layer {layer_name}: "
                         f"activation {activation.shape}, mean_representation {mean_representation.shape}")
    
    activation_flat = activation.view(activation.size(0), -1) 
    mean_flat = mean_representation.view(mean_representation.size(0), -1) 
    
    diff = torch.abs(activation_flat - mean_flat) 
    sum_abs = torch.abs(activation_flat) + torch.abs(mean_flat) + 1e-8  
    canberra_channelwise = torch.sum(diff / sum_abs, dim=1)
    
    canberra_distance = torch.mean(canberra_channelwise).item()
    
    return canberra_distance


def compute_canberra_distance_for_all_layers(activations, mean_representations, model):
    all_canberra_distances = []

    for name, layer in model.named_modules():
        if layer in activations:
            activation = activations[layer]

            if name not in mean_representations:
                raise KeyError(f"Layer {name} not found in mean_representations. "
                               f"Available keys: {list(mean_representations.keys())}")

            mean_representation = mean_representations[name]

            canberra_distance = compute_canberra_distance_layer(activation, mean_representation, name)
            all_canberra_distances.append(canberra_distance)
    
    return all_canberra_distances

def load_image_from_path(image_path, device):
    image = np.load(image_path)
    
    image_tensor = torch.tensor(image, dtype=torch.float32)
    
    if image_tensor.ndimension() == 2:
        image_tensor = image_tensor.unsqueeze(0).repeat(3, 1, 1)
    
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image_tensor = transform(image_tensor)
    
    return image_tensor.to(device)

def generate_dataset_for_svm(successful_adversarial_images_fgsm, train_loader, model, mean_representations, device):
    features = []
    labels = []

    hooks, activations = register_hooks_for_layers(model)

    # Process adversarial images
    print("Processing adversarial images...")
    for item in tqdm(successful_adversarial_images_fgsm, desc="Adversarial Images", unit="image"):
        adv_image_path = item['adv_image_path']
        normal_image_path = item['original_image_path']
        
        adv_image = load_image_from_path(adv_image_path, device)
        normal_image = load_image_from_path(normal_image_path, device)

        with torch.no_grad():
            model(adv_image.unsqueeze(0))
        # Compute Canberra distances for the adversarial image across all layers
        adv_canberra_distances = compute_canberra_distance_for_all_layers(activations, mean_representations, model)
        features.append(adv_canberra_distances)
        labels.append(1)  # Adversarial image label

        with torch.no_grad():
            model(normal_image.unsqueeze(0))

        # Compute Canberra distances for the normal image across all layers
        normal_canberra_distances = compute_canberra_distance_for_all_layers(activations, mean_representations, model)
        features.append(normal_canberra_distances)
        labels.append(0)  # Normal image label

    # Process normal images from train_loader
    print("Processing normal images...")
    for images, _ in tqdm(train_loader, desc="Normal Images", unit="batch"):
        images = images.to(device)
        for image in images:
            # Perform forward pass for the normal image to get activations
            with torch.no_grad():
                model(image.unsqueeze(0))  # Add batch dimension and pass through the model

            # Compute Canberra distances for the normal image across all layers
            normal_canberra_distances = compute_canberra_distance_for_all_layers(activations, mean_representations, model)
            features.append(normal_canberra_distances)
            labels.append(0)  # Normal image label
            
            if len(features) == len(successful_adversarial_images_fgsm) * 2:  # 2 per adversarial image
                break
        if len(features) == len(successful_adversarial_images_fgsm) * 2:
            break

    for hook in hooks:
        hook.remove()

    features = np.array(features)
    labels = np.array(labels)

    return features, labels

features, labels = generate_dataset_for_svm(metadata_list, train_loader, model, mean_representations, device)
# print(f"Features shape: {features.shape}, Labels shape: {labels.shape}")
print("SVM dataset generated successfully!")


features_flattened = np.array([feature.flatten() for feature in features])

features_df = pd.DataFrame(features_flattened)

features_df['label'] = labels

features_df.to_csv('dataset_for_svm.csv', index=False)

print("Features and labels have been saved to 'dataset_for_svm.csv'.")


dataset = pd.read_csv('dataset_for_svm.csv')
dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
X = dataset.drop(columns=['label']).values  # Features
y = dataset['label'].values  # Labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# print(y_train[:20])

df = pd.read_csv('dataset_for_svm.csv')

X = df.drop('label', axis=1).values  # Assuming the label column is named 'label'
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svm_classifier = SVC(kernel='linear', random_state=42)

# Train the SVM model
print("Training the SVM classifier...")
svm_classifier.fit(X_train, y_train)
print("Training complete!")

y_pred = svm_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

####################################### MITIGATION #########################################

activations = {}

def hook_fn(module, input, output):
    activations[module] = output.detach()

def register_hooks(model):
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Conv2d):
            layer.register_forward_hook(hook_fn)

def compute_epsilon_from_folder(model, adv_data, device):
    epsilon_values = {}

    model.eval()
    model.to(device)

    for data in tqdm(adv_data, desc="Computing epsilon_ij"):
        original_image_path = data['original_image_path']
        adv_image_path = data['adv_image_path']
        
        original_image = load_image_from_path(original_image_path, device)
        adv_image = load_image_from_path(adv_image_path, device)

        activations.clear()

        model(original_image.unsqueeze(0))
        activations_original = activations.copy()

        activations.clear()
        model(adv_image.unsqueeze(0))
        activations_adv = activations.copy()

        for layer in activations_original.keys():
            original_act = activations_original[layer]
            adv_act = activations_adv[layer]

            epsilon = torch.abs(adv_act - original_act).mean(dim=(0, 2, 3)) 
            if layer not in epsilon_values:
                epsilon_values[layer] = []
            epsilon_values[layer].append(epsilon.cpu().numpy())

    return epsilon_values

register_hooks(model)
epsilon_values = compute_epsilon_from_folder(model, metadata_list, device)

# Recitifier model

# Model setup
mitigation_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

mitigation_model.classifier[5] = nn.Dropout(p=0.5)
mitigation_model.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes, bias=True)

for param in mitigation_model.features[:-8].parameters():
    param.requires_grad = False
for param in mitigation_model.features[-8:].parameters():
    param.requires_grad = True

mitigation_model = mitigation_model.to(device)

state_dict = torch.load("./vgg16_model.pth")
mitigation_model.load_state_dict(state_dict)

print("Model weights successfully loaded!")

def apply_selective_dropout(model, epsilon_values, eta=3, kappa=0.2):
    layer_aggregated_epsilon = {
        layer: np.mean(np.abs(epsilon), axis=0).sum()
        for layer, epsilon in epsilon_values.items()
    }

    sorted_layers = sorted(layer_aggregated_epsilon.items(), key=lambda x: x[1], reverse=True)
    top_layers = sorted_layers[:eta]
    for layer_name, _ in top_layers:
        if layer_name not in model.state_dict():
            continue
        layer = dict(model.named_modules())[layer_name]
        
        if isinstance(layer, torch.nn.Conv2d):
            weights = layer.weight.data
            num_filters = weights.size(0)

            filter_epsilon = np.mean(np.abs(epsilon_values[layer_name]), axis=(1, 2, 3))  # Mean over H and W
            top_filters = np.argsort(filter_epsilon)[-int(kappa * num_filters):]

            for idx in top_filters:
                weights[idx] = 0

            layer.weight.data = weights

    return model


def load_image_for_model(image_path, device):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (224, 224))

    image_tensor = torch.tensor(image, dtype=torch.float32) / 255.0  # Normalize to [0, 1]
    
    image_tensor = image_tensor.permute(2, 0, 1)

    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor


def mitigate(mitigation_model, metadata_list, device):
    correct_predictions = 0
    total_images = len(metadata_list)

    for data in tqdm(metadata_list, desc="Mitigating images"):
        adv_image_tensor = load_image_from_path(data['adv_image_path'], device)
        
        # Apply median filtering for denoising
        adv_image_tensor_np = adv_image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        denoised_adv_image = cv2.medianBlur((adv_image_tensor_np * 255).astype(np.uint8), 1)  # Apply 5x5 median filter
        denoised_adv_image_tensor = torch.tensor(denoised_adv_image, dtype=torch.float32) / 255.0
        denoised_adv_image_tensor = denoised_adv_image_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

        # Step 2: Predict class for denoised image using mitigation model
        with torch.no_grad():
            mitigated_class = mitigation_model(denoised_adv_image_tensor).argmax(dim=1).item()
        

        # Step 3: Compare mitigated prediction with the original class
        if mitigated_class == data['original_label']:
            correct_predictions += 1

    # Calculate accuracy
    accuracy = correct_predictions / total_images
    return accuracy

accuracy = mitigate(mitigation_model, metadata_list, device)
print(f"Mitigation Accuracy withot ANF: {accuracy * 100:.2f}%")

# Adaptive Noise Reduction

def smooth_image(image, filter_type='gaussian'):
    if filter_type == 'gaussian':
        return gaussian_filter(image, sigma=1)
    elif filter_type == 'median':
        return median_filter(image, size=3)

def quantize_image(image, levels):
    interval = 256 // levels
    quantized_image = (image // interval) * interval
    return quantized_image  

def calculate_shannon_entropy(image):
    flattened_image = image.flatten()
    
    hist, _ = np.histogram(flattened_image, bins=256, range=(0, 256))
    
    probabilities = hist / np.sum(hist)
    
    probabilities = probabilities[probabilities > 0]
    
    entropy = -np.sum(probabilities * np.log2(probabilities))
    
    return entropy

def adaptive_noise_filter(image):
    entropy = calculate_shannon_entropy(image)
    
    if entropy < 4.0:
        levels = 2
        smoothing_needed = False
    elif 4.0 <= entropy <= 5.0:
        levels = 4
        smoothing_needed = False
    else:
        levels = 6
        smoothing_needed = True

    quantized_image = quantize_image(image, levels)

    if smoothing_needed:
        smoothed_image = smooth_image(quantized_image, filter_type='gaussian')
    else:
        smoothed_image = quantized_image

    output_image = np.where(
        np.abs(image - quantized_image) <= np.abs(image - smoothed_image),
        quantized_image,
        smoothed_image
    )

    return output_image

def detect_adversarial_using_noise_reduction(model, metadata_list, device):
    correct_predictions = 0
    total_images = 0

    model.eval()
    model.to(device)

    for data in tqdm(metadata_list, desc="Processing images"):
        # Load the image
        original_image_path = data['original_image_path']
        actual_class = data['original_label']

        image = cv2.imread(original_image_path)
        if image is None:
            continue
        image = cv2.resize(image, (224, 224))
        image_tensor = torch.tensor(image.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0

        # Apply adaptive noise reduction
        denoised_image = adaptive_noise_filter(image)
        denoised_tensor = torch.tensor(denoised_image.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0

        # Get prediction from the denoised image
        with torch.no_grad():
            denoised_prediction = model(denoised_tensor).argmax(dim=1).item()

        # Compare prediction with actual class to detect adversarial image
        if denoised_prediction != data['adv_label']:
            correct_predictions += 1

        total_images += 1

    # Calculate detection accuracy
    accuracy = correct_predictions / total_images if total_images > 0 else 0
    return accuracy

accuracy = detect_adversarial_using_noise_reduction(model, metadata_list, device)
print(f"Detection Accuracy of ANF: {accuracy * 100:.2f}%")

def mitigate2(mitigation_model, metadata_list, device):
    correct_predictions = 0
    total_images = len(metadata_list)

    for data in tqdm(metadata_list, desc="Mitigating images"):
        adv_image_tensor = load_image_from_path(data['adv_image_path'], device)
        
        adv_image_tensor_np = adv_image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        # Apply adaptive noise filtering for denoising

        denoised_adv_image = adaptive_noise_filter(adv_image_tensor_np)
        denoised_adv_image_tensor = torch.tensor(denoised_adv_image, dtype=torch.float32) / 255.0
        denoised_adv_image_tensor = denoised_adv_image_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

        # Step 2: Predict class for denoised image using mitigation model
        with torch.no_grad():
            mitigated_class = mitigation_model(denoised_adv_image_tensor).argmax(dim=1).item()
        

        # Step 3: Compare mitigated prediction with the original class
        if mitigated_class == data['original_label']:
            correct_predictions += 1

    accuracy = correct_predictions / total_images
    return accuracy


accuracy = mitigate2(model, metadata_list, device)

print(f"Mitigation Accuracy with ANF denoising: {accuracy * 100:.2f}%")

