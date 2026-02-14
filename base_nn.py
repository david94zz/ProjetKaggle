import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
import geopandas as gpd

warnings.filterwarnings('ignore')

# 1. Configuration and Mappings
TARGET_MAP = {
    'Demolition': 0, 'Road': 1, 'Residential': 2, 
    'Commercial': 3, 'Industrial': 4, 'Mega Projects': 5
}

# Only use the 15 raw RGB mean features
FEATURE_COLS = [
    'img_red_mean_date1', 'img_blue_mean_date1', 'img_green_mean_date1',
    'img_red_mean_date2', 'img_blue_mean_date2', 'img_green_mean_date2',
    'img_red_mean_date3', 'img_blue_mean_date3', 'img_green_mean_date3',
    'img_red_mean_date4', 'img_blue_mean_date4', 'img_green_mean_date4',
    'img_red_mean_date5', 'img_blue_mean_date5', 'img_green_mean_date5'
]

def main():
    print("Loading data for Neural Network Baseline...")
    
    # We will use geopandas just to read it properly like in your main script
    
    train_df = gpd.read_file('train.geojson')
    test_df = gpd.read_file('test.geojson')

    # 2. Data Preparation
    # Drop NaNs only from training data (Never drop from test data!)
    train_df_clean = train_df.dropna(subset=FEATURE_COLS + ['change_type'])
    
    # Extract features and target
    X_train_raw = train_df_clean[FEATURE_COLS].values
    y_train_raw = train_df_clean['change_type'].map(TARGET_MAP).values
    
    # Extract test features and keep IDs for submission
    X_test_raw = test_df[FEATURE_COLS].fillna(0).values # Fillna instead of dropna for test set!
    test_ids = test_df['index']

    # 3. Convert to PyTorch Tensors and Normalize (divide by 255.0 for RGB)
    X_train_tensor = torch.tensor(X_train_raw, dtype=torch.float32) / 255.0
    y_train_tensor = torch.tensor(y_train_raw, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test_raw, dtype=torch.float32) / 255.0

    print(f"Training Tensor Shape: {X_train_tensor.shape}")
    print(f"Target Tensor Shape: {y_train_tensor.shape}")

    # 4. Define the Simple Neural Network Architecture
    class SimpleNN(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            out = self.fc1(x)
            out = self.relu(out)
            out = self.fc2(out)
            return out

    # 5. Initialize Model, Loss, and Optimizer
    input_size = 15
    hidden_size = 5
    output_size = 6
    
    model = SimpleNN(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # 6. Training Loop
    num_epochs = 10000
    print("Starting Training Loop...")
    for epoch in range(num_epochs):
        model.train()
        
        # Forward pass
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 1000 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # 7. Evaluation on Test Set and Submission
    print("Generating predictions for Kaggle...")
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        # Get the index of the max log-probability
        predicted_classes = torch.argmax(test_outputs, dim=1).numpy()

    # Create submission file correctly mapped to Test IDs
    submission_df = pd.DataFrame({
        'Id': test_ids,
        'change_type': predicted_classes
    })
    
    submission_df.to_csv("baseline_nn_submission.csv", index=False)
    print("Saved 'baseline_nn_submission.csv'")

if __name__ == "__main__":
    main()