import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from utils import to_device  # Assume this is in your utils
from models import ecgTransForm
import os
from matplotlib.colors import ListedColormap

# Define the function for t-SNE visualization
def tsne_visualization(model, data_loader, device, save_dir, class_names):
    # Prepare model for inference
    model.to(device).eval()

    # Lists to store features and corresponding labels
    all_features = []
    all_labels = []

    # Extract features from the data loader
    with torch.no_grad():
        for batches in data_loader:
            batches = to_device(batches, device)
            data = batches['samples'].float()
            labels = batches['labels'].long()

            # Get model output features (before final softmax or decision layer)
            logits = model(data)
            features = logits.detach().cpu().numpy()

            all_features.append(features)
            all_labels.append(labels.detach().cpu().numpy())

    # Concatenate all features and labels
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    reduced_features = tsne.fit_transform(all_features)

    # Log t-SNE results (for debugging)
    print("t-SNE reduced features shape:", reduced_features.shape)
    print("First 10 reduced features:", reduced_features[:10])

    # Label encoding for color map
    le = LabelEncoder()
    all_labels_encoded = le.fit_transform(all_labels)

    # Enhanced color palette with more contrast
    enhanced_colors = ['#FF8A80', '#FFB74D', '#FFEB3B', '#66BB6A', '#4FC3F7']  # Vibrant Morandi-inspired colors
    custom_cmap = ListedColormap(enhanced_colors)

    # Create the t-SNE plot
    plt.figure(figsize=(10, 8))
    # Adjust the size of data points using the `s` parameter (smaller value, smaller points)
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=all_labels_encoded, cmap=custom_cmap, s=1)

    # Create legend with class names
    handles, _ = scatter.legend_elements(prop='colors')
    plt.legend(handles, class_names, title="Classes")
    plt.colorbar(scatter)
    plt.title('t-SNE Visualization of ECG Features on PhysioNet 2017')

    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')

    # Ensure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  # Create the directory if it doesn't exist

    # Save the plot
    save_path = os.path.join(save_dir, "physionet3.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"t-SNE visualization saved at: {save_path}")


# Test script to run the t-SNE visualization after training
def run_tsne_visualization():
    # Load the best model checkpoint
    model_checkpoint_path = r"D:\Na\GT\dmo5-py2017\experiments_logs\py2017-epoch-all-200\run1_22_18\checkpoint_best.pt"
    checkpoint = torch.load(model_checkpoint_path)

    # Load your dataset (replace with actual dataset loading code)
    from dataloader import data_generator
    from configs.data_configs import get_dataset_class
    from configs.hparams import get_hparams_class

    dataset = 'py20172'  # Choose the dataset (can be 'mit' or 'ptb')
    dataset_class = get_dataset_class(dataset)
    dataset_configs = dataset_class()

    hparams_class = get_hparams_class("supervised")
    hparams = hparams_class().train_params

    # Load the data loader for test dataset
    data_path = r'D:\Na\dataset'  # Update with actual data path

    # Modify the data generator call to use 50% of the test set
    test_dl, _, _, _ = data_generator(data_path, dataset, hparams, test_subset=True)

    # Create model instance and load the saved state_dict
    model = ecgTransForm(configs=dataset_configs, hparams=hparams)
    model.load_state_dict(checkpoint["model"])

    # Set device (ensure it matches your training device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define class names for MIT dataset
    class_names = ['A', 'N', 'O', '~']  # MIT dataset has 5 classes

    # Run t-SNE visualization
    save_dir = 'keshi'  # Specify where to save the visualization
    tsne_visualization(model, test_dl, device, save_dir, class_names)


if __name__ == "__main__":
    run_tsne_visualization()
