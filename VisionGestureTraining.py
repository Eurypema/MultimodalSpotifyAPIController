import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import psutil

npy_dir = "H:\\Training Data"
chunk_dir = "H:\\Training Data\\chunks"

chunk_size = 100  # Number of samples per chunk

# Function to split data into chunks and save them
def save_data_chunks(npy_dir, chunk_dir, chunk_size):
    file_paths = [os.path.join(npy_dir, fname) for fname in os.listdir(npy_dir) if fname.endswith('.npy')]
    for file_idx, file_path in enumerate(tqdm(file_paths, desc="Processing files")):
        data = np.load(file_path, mmap_mode='r')
        num_chunks = int(np.ceil(data.shape[0] / chunk_size))
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, data.shape[0])
            chunk_data = data[start_idx:end_idx]
            chunk_fname = f"{os.path.basename(file_path).split('.')[0]}_chunk{chunk_idx}.npy"
            np.save(os.path.join(chunk_dir, chunk_fname), chunk_data)

# Check if the chunk directory already contains chunked data
if not os.path.exists(chunk_dir):
    os.makedirs(chunk_dir)

if not any(fname.endswith('.npy') for fname in os.listdir(chunk_dir)):
    print("No chunked data found, starting chunking process...")
    save_data_chunks(npy_dir, chunk_dir, chunk_size)
else:
    print("Chunked data already exists, skipping chunking process.")

# Custom collate function
def custom_collate_fn(batch):
    data, labels = zip(*batch)
    data = torch.cat(data)
    labels = torch.tensor(labels)
    return data, labels

# Custom Dataset class for large npy files with chunking
class ChunkedDataset(Dataset):
    def __init__(self, chunk_dir, labels, target_shape):
        self.chunk_dir = chunk_dir
        self.labels = labels
        self.chunk_files = [os.path.join(chunk_dir, fname) for fname in os.listdir(chunk_dir) if fname.endswith('.npy')]
        self.target_shape = target_shape
        print(f"Initialized ChunkedDataset with {len(self.chunk_files)} chunk files, target shape: {target_shape}, chunk size: {chunk_size}")

    def __len__(self):
        return len(self.chunk_files)

    def __getitem__(self, idx):
        chunk_file = self.chunk_files[idx]
        label = self.labels[idx // (len(self.chunk_files) // len(self.labels))]
        data = np.load(chunk_file)
        data = torch.tensor(data, dtype=torch.float32)

        # Pad data if necessary
        if data.shape != self.target_shape:
            padded_data = torch.zeros((data.shape[0],) + self.target_shape, dtype=torch.float32)
            slices = (slice(0, data.shape[0]),) + tuple(slice(0, min(s1, s2)) for s1, s2 in zip(data.shape[1:], self.target_shape))
            padded_data[slices] = data
            data = padded_data

        print(f"Loaded chunk {idx + 1}/{len(self.chunk_files)} from file {chunk_file}")
        print(f"Data shape: {data.shape}")
        return data, torch.tensor(label, dtype=torch.long)

def log_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"Memory usage: {mem_info.rss / (1024 ** 3):.2f} GB")  # Resident Set Size

if __name__ == "__main__":  # Need to guard the entry point of the program to avoid creating new processes incorrectly
    # Step 0: Make sure CUDA works
    print("CUDA Available: ", torch.cuda.is_available())
    print("CUDA Version: ", torch.version.cuda)
    print("cuDNN Version: ", torch.backends.cudnn.version())
    print("Device Name: ", torch.cuda.get_device_name(torch.cuda.current_device()))

    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Step 1: Read the Logged Data
    # Initialize an empty list to hold data and labels
    all_data = []
    all_labels = []

    # Function to map file names to labels
    def label_from_filename(filename):
        if 'wave' in filename:
            return 0
        elif 'push' in filename:
            return 1
        elif 'pull' in filename:
            return 2
        else:
            return -1

    # Create a list of file paths and corresponding labels
    file_paths = [os.path.join(npy_dir, fname) for fname in os.listdir(npy_dir) if fname.endswith('.npy')]
    labels = [label_from_filename(fname) for fname in os.listdir(npy_dir) if fname.endswith('.npy')]

    # Step 2: Preprocessing

    # Calculate class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)  # Move class weights to GPU

    # Split the data into training and validation sets
    train_paths, val_paths, y_train, y_val = train_test_split(file_paths, labels, test_size=0.2, random_state=42)

    # Determine the maximum shape of the data
    max_shape = [0, 0, 0, 0]
    for file_path in file_paths:
        data = np.load(file_path, mmap_mode='r')
        max_shape = [max(max_shape[i], data.shape[i]) for i in range(len(data.shape))]

    print(f"Max shape determined: {max_shape}")

    # Create the dataset and dataloader
    train_dataset = ChunkedDataset(chunk_dir, y_train, target_shape=(3, 224, 224))
    val_dataset = ChunkedDataset(chunk_dir, y_val, target_shape=(3, 224, 224))

    # Create DataLoaders for training and validation with reduced batch size
    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=2, collate_fn=custom_collate_fn)  # Further reduced batch size and workers
    val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=2, collate_fn=custom_collate_fn)  # Further reduced batch size and workers

    # Step 3: Build and Train a PyTorch Model
    # Define a sequential model for gesture recognition
    class VisionRecognitionModel(nn.Module):
        def __init__(self):
            super(VisionRecognitionModel, self).__init__()
            self.conv_layers = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # Adjust input channels to 3
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout(p=0.5)  # Dropout for regularization
            )
            self.fc_layers = nn.Sequential(
                nn.Flatten(),
                nn.Linear(32 * 56 * 56, 256),
                nn.ReLU(),
                nn.Dropout(p=0.5),  # Dropout for regularization
                nn.Linear(256, 3)  # 3 classes: push, pull, wave
            )

        def forward(self, x):
            x = self.conv_layers(x)
            x = self.fc_layers(x)
            return x

    model = VisionRecognitionModel().to(device)  # Initialize model and move to GPU

    # Initialize the model, loss function, and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

    # Early stopping parameters
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    # Initialize lists to store metrics for each epoch
    precision_list = []
    recall_list = []
    f1_list = []

    # Training loop
    for epoch in range(30):  # Number of epochs
        model.train()  # Set model to training mode
        for features, labels in train_loader:
            optimizer.zero_grad()
            features, labels = features.to(device), labels.to(device)  # Move data to GPU
            print(f"Features shape: {features.shape}, Labels shape: {labels.shape}")  # Debug statement
            log_memory_usage()  # Log memory usage
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation phase
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)  # Move data to GPU
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = accuracy_score(all_labels, all_preds)

        # Calculate precision, recall, and F1 score
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')
        cm = confusion_matrix(all_labels, all_preds)

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

        print(f"Epoch {epoch + 1}, Training Loss: {loss.item()}, Validation Loss: {avg_val_loss}, Validation Accuracy: {val_accuracy}")
        print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
        print(f"Confusion Matrix:\n{cm}")

        # Check for early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping")
            break

    # Save the model
    torch.save(model.state_dict(), 'visual_gesture_recognition_model.pth')
    print("Model saved to visual_gesture_recognition_model.pth")
