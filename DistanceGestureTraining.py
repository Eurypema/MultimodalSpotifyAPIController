# Import necessary libraries
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import io

# Step 0: Make sure CUDA works
print("CUDA Available: ", torch.cuda.is_available()) 
print("CUDA Version: ", torch.version.cuda)
print("cuDNN Version: ", torch.backends.cudnn.version())
print("Device Name: ", torch.cuda.get_device_name(torch.cuda.current_device()))

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {device}")

# Step 1: Read the Logged Data
# Define the directory path containing the log files
log_dir = "C:\\Users\\gordo\\AppData\\Local\\teraterm5"

# Initialize an empty list to hold dataframes
all_data = []

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

# Iterate through each file in the directory
for file_name in os.listdir(log_dir):
    if file_name.endswith('.log'):
        file_path = os.path.join(log_dir, file_name)
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        cleaned_lines = []
        for i, line in enumerate(lines):
            try:
                pd.read_csv(io.StringIO(line), names=['timestamp', 'distance'])
                cleaned_lines.append(line)
            except pd.errors.ParserError:
                print(f"Error parsing file {file_name} at line {i+1}: {line.strip()}")
        
        # If there are cleaned lines, process them
        if cleaned_lines:
            try:
                data = pd.read_csv(io.StringIO('\n'.join(cleaned_lines)), names=['timestamp', 'distance'])
                data['target'] = label_from_filename(file_name)
                all_data.append(data)
            except pd.errors.ParserError as e:
                print(f"Error parsing cleaned lines in file {file_name}: {e}")

# Combine all dataframes into one
if all_data:
    try:
        data = pd.concat(all_data, ignore_index=True)
    except pd.errors.ParserError as e:
        print(f"Error concatenating dataframes: {e}")

# Step 2: Preprocess the Data
# Convert distance to numeric, replacing errors with NaN
data['distance'] = pd.to_numeric(data['distance'], errors='coerce')
# Drop rows with any NaN values
data.dropna(inplace=True)

# Calculate the time difference
data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
data['time_diff'] = data['timestamp'].diff().dt.total_seconds().fillna(0)

# Calculate the rate of change of distance (delta_distance)
data['delta_distance'] = data['distance'].diff().fillna(0)


# Drop rows with any NaN values again after new features
data.dropna(inplace=True)

# Ensure all columns are numeric
data[['distance', 'delta_distance', 'time_diff']] = data[['distance', 'delta_distance', 'time_diff']].apply(pd.to_numeric, errors='coerce')

# Normalize the data (including the new features)
data[['distance', 'delta_distance', 'time_diff']] = (data[['distance', 'delta_distance', 'time_diff']] - data[['distance', 'delta_distance', 'time_diff']].mean()) / data[['distance', 'delta_distance', 'time_diff']].std()

# Filter out any rows with undefined labels
data = data[data['target'] != -1]

# Extract features (distance, delta_distance, time_diff) and labels (target)
X = data[['distance', 'delta_distance', 'time_diff']].values.astype(np.float32)
y = data['target'].values

# Data augmentation
def augment_data(X, y):
    augmented_X, augmented_y = [], []
    for i in range(len(X)):
        original = X[i]
        label = y[i]
        augmented_X.append(original)
        augmented_y.append(label)
        
        # Add Gaussian noise
        noise = np.random.normal(0, 0.01, original.shape)
        augmented_X.append(original + noise)
        augmented_y.append(label)
        
        # Scale data
        scale_factor = np.random.uniform(0.9, 1.1)
        augmented_X.append(original * scale_factor)
        augmented_y.append(label)
        
        # Time shift
        shift = np.random.randint(1, 5)
        shifted = np.roll(original, shift)
        augmented_X.append(shifted)
        augmented_y.append(label)
        
        # Flip
        flipped = np.flip(original)
        augmented_X.append(flipped)
        augmented_y.append(label)
    
    return np.array(augmented_X), np.array(augmented_y)

X_augmented, y_augmented = augment_data(X, y)

# Calculate class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)  # Move class weights to GPU

# Convert features and labels to PyTorch tensors
X_tensor = torch.tensor(X_augmented, dtype=torch.float32)  # Do not move to GPU yet
y_tensor = torch.tensor(y_augmented, dtype=torch.long)     # Do not move to GPU yet

# Step 4: Define a Dataset and DataLoader
# Custom Dataset class for sensor data
class SensorDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        # Return a tuple of features and corresponding label
        return self.features[idx], self.labels[idx]

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# Create DataLoaders for training and validation
train_loader = DataLoader(SensorDataset(X_train, y_train), batch_size=32, shuffle=True, pin_memory=False)
val_loader = DataLoader(SensorDataset(X_val, y_val), batch_size=32, shuffle=False, pin_memory=False)

# Step 5: Build and Train a PyTorch Model
# LSTM-based model for gesture recognition
class GestureRecognitionModel(nn.Module): # nn.Module is the base class for all neural network modules in PyTorch
    def __init__(self, input_size, hidden_size, output_size): # Initializes network architecture; 
        super(GestureRecognitionModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=5, batch_first=True) # Defines LSTM layer. batch_first indicates input and output tensors are given as (batch, sequence, feature)
        self.dropout = nn.Dropout(0.5) # Prevents overfitting by randomly setting half of the activations to zero during training
        
        # Defines fully-connected linear layers
        self.fc1 = nn.Linear(hidden_size, hidden_size) 
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.bn = nn.BatchNorm1d(hidden_size) # Defines batch normalization layer to normalize LSTM layer output
        
    def forward(self, x): # Forward pass, computes output given input.
        lstm_out, _ = self.lstm(x) # Passes the input through the LSTM layer
        lstm_out = self.dropout(lstm_out) # Applies dropout to the LSTM output

        # Extracts the output corresponding to the last time step for each time step. Last hidden state contains most relevant information for sequence prediction.
        lstm_out = lstm_out[:, -1, :]
 
        lstm_out = self.bn(lstm_out) # Batch normalization
        fc1_out = self.fc1(lstm_out) # Perculation through first linear layer
        fc1_out = nn.ReLU()(fc1_out) # ReLU=max(0,x); introduces nonlinearity which is needed for approximating any continuous function
        fc2_out = self.fc2(fc1_out) # Perculation through second linear layer
        fc2_out = nn.ReLU()(fc2_out)
        output = self.fc3(fc2_out)  # Perculation through third linear layer
        return output

# Initialize the model, loss function, and optimizer
input_size = 3  # Three input features: distance, delta_distance, time_diff
hidden_size = 100  # Number of LSTM hidden units
output_size = 3  # Number of output classes: 'wave', 'push', 'pull'

model = GestureRecognitionModel(input_size, hidden_size, output_size).to(device)  # Move model to GPU

# Load only the model weights
# model.load_state_dict(torch.load('gesture_recognition_model.pth', map_location=device, weights_only=True))

# Measures the dissimilarity between predicted probability distribution and actual distribution, taking weights of classes as argument.
criterion = nn.CrossEntropyLoss(weight=class_weights) 

# Set up for optimization algorithm
optimizer = optim.Adam(model.parameters(), lr=0.00005, weight_decay=1e-5)
# Uses the Adaptive Moment Estimation optization algorithm to continually optimize parameters to best fit training data
# Low learning rate for small updates to model parameters
# weight decay adds a small regularization term to the loss function and is used to discourage overly complex models adn large weights and thus prevent overfitting

# Early stopping parameters
best_val_loss = float('inf')
patience = 5
patience_counter = 0

# Initialize lists to store metrics for each epoch
precision_list = []
recall_list = []
f1_list = []

for epoch in range(30):  # Number of epochs
    model.train()  # Set model to training mode
    for features, labels in train_loader:  # Iterate over the training data
        optimizer.zero_grad()  # Clear the gradients from the previous step
        features, labels = features.to(device), labels.to(device)  # Move data to GPU
        features = features.unsqueeze(1)  # Add a dimension for single-channel input
        outputs = model(features)  # Forward pass: compute model output
        loss = criterion(outputs, labels)  # Compute the loss function
        loss.backward()  # Backward pass: compute gradients
        optimizer.step()  # Update model parameters based on gradients
    
    # Validation phase
    model.eval()  # Set model to evaluation mode (disables dropout, etc.)
    val_loss = 0  # Initialize validation loss
    all_preds = []  # List to store all predicted labels
    all_labels = []  # List to store all true labels
    with torch.no_grad():  # Disable gradient calculation (faster and saves memory)
        for features, labels in val_loader:  # Iterate over the validation data
            features, labels = features.to(device), labels.to(device)  # Move data to GPU
            features = features.unsqueeze(1)  # Add a dimension for single-channel input
            outputs = model(features)  # Forward pass: compute model output
            loss = criterion(outputs, labels)  # Compute the loss function
            val_loss += loss.item()  # Accumulate the validation loss
            
            _, preds = torch.max(outputs, 1)  # Get the predicted class (highest probability)
            all_preds.extend(preds.cpu().numpy())  # Store predictions, convert to numpy
            all_labels.extend(labels.cpu().numpy())  # Store true labels, convert to numpy
    
    avg_val_loss = val_loss / len(val_loader)  # Compute average validation loss
    val_accuracy = accuracy_score(all_labels, all_preds)  # Calculate validation accuracy
    
    # Calculate precision, recall, and F1 score
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)  # Compute the confusion matrix
    
    precision_list.append(precision)  # Append precision to list
    recall_list.append(recall)  # Append recall to list
    f1_list.append(f1)  # Append F1 score to list
    
    print(f"Epoch {epoch + 1}, Training Loss: {loss.item()}, Validation Loss: {avg_val_loss}, Validation Accuracy: {val_accuracy}")
    print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
    print(f"Confusion Matrix:\n{cm}")
    
    # Check for early stopping
    if avg_val_loss < best_val_loss:  # Check if current validation loss is the best
        best_val_loss = avg_val_loss  # Update best validation loss
        patience_counter = 0  # Reset patience counter
    else:
        patience_counter += 1  # Increment patience counter if no improvement
    
    if patience_counter >= patience:  # Stop if no improvement for a certain number of epochs
        print("Early stopping")
        break  # Exit the training loop


# Save the model
torch.save(model.state_dict(), 'gesture_recognition_model.pth')
print("Model saved to gesture_recognition_model.pth")