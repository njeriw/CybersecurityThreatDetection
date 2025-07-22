'''
This script serves as the main entry point for training and hyperparameter tuning of a neural network classifier for cyber threat detection. 
'''

'''
Importing libraries
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from skorch import NeuralNetClassifier
from skopt.space import Real, Integer, Categorical
from skopt import BayesSearchCV


def load_data():
    """Load the cybersecurity datasets with proper error handling."""
    try:
        train_df = pd.read_csv('cybersecurity_train.csv')
        test_df = pd.read_csv('cybersecurity_test.csv')
        val_df = pd.read_csv('cybersecurity_val.csv')
        print("Data loaded successfully")
        return train_df, test_df, val_df
    except FileNotFoundError as e:
        raise FileNotFoundError(f'Unable to load CSV file: {e}')


class NeuralNetwork(nn.Module):
    def __init__(self, input_features, hidden_size=128, num_layers=2, 
                 dropout_rate=0.5, activation_function_name='relu'):
        super().__init__()
        
        activation_map = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'tanh': nn.Tanh()
        }
        
        self.activation = activation_map.get(activation_function_name, nn.ReLU())
        
        layers = []
        
        layers.extend([
            nn.Linear(input_features, hidden_size),
            self.activation,
            nn.Dropout(dropout_rate)
        ])
        
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_size, hidden_size // 2),
                self.activation,
                nn.Dropout(dropout_rate)
            ])
            hidden_size = hidden_size // 2
        
        layers.append(nn.Linear(hidden_size, 1))
        
        self.network = nn.Sequential(*layers)
        self._initialize_weights()
    
    def forward(self, x):
        return self.network(x)
    
    def _initialize_weights(self):
        """Initialize weights using He initialization for ReLU networks."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    init.zeros_(module.bias)

def preprocess_data(train_df, test_df, val_df):
    X_train = train_df.drop('sus_label', axis=1)
    y_train = train_df['sus_label']
    X_test = test_df.drop('sus_label', axis=1)
    y_test = test_df['sus_label']
    X_val = val_df.drop('sus_label', axis=1)
    y_val = val_df['sus_label']
    

    numerical_columns = X_train.select_dtypes(include=[np.number]).columns
    

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_columns)
        ],
        remainder='passthrough'  
    )
    
    X_train_processed = preprocessor.fit_transform(X_train).astype(np.float32)
    X_test_processed = preprocessor.transform(X_test).astype(np.float32)
    X_val_processed = preprocessor.transform(X_val).astype(np.float32)
    
    return X_train_processed, X_test_processed, X_val_processed, y_train, y_test, y_val, preprocessor

def create_data_loaders(X_train, X_test, X_val, y_train, y_test, y_val, batch_size=64):
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, val_loader

def train_model(model, train_loader, val_loader, epochs=10, lr=1e-3, weight_decay=1e-5):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        if (epoch + 1) % 2 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    
    return train_losses, val_losses

def evaluate_model(model, test_loader):
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            predicted = (probs > 0.5).float()
            
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(true_labels, predictions)
    return accuracy


class SkorchNeuralNetwork(nn.Module):
    def __init__(self, input_features, hidden_size=128, num_layers=2, 
                 dropout_rate=0.5, activation_function_name='relu'):
        super().__init__()
        
        activation_map = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'tanh': nn.Tanh()
        }
        
        self.activation = activation_map.get(activation_function_name, nn.ReLU())

        layers = []

        layers.extend([
            nn.Linear(input_features, hidden_size),
            self.activation,
            nn.Dropout(dropout_rate)
        ])

        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_size, hidden_size // 2),
                self.activation,
                nn.Dropout(dropout_rate)
            ])
            hidden_size = hidden_size // 2
        
        layers.append(nn.Linear(hidden_size, 1))
        
        self.network = nn.Sequential(*layers)
        self._initialize_weights()
    
    def forward(self, x):
        return self.network(x).squeeze(-1) 
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    init.zeros_(module.bias)

def tune_classifier(X_train, y_train, cv=5, n_iter=50):
    
    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    

    input_features = X_train.shape[1]
    
    net = NeuralNetClassifier(
        module=SkorchNeuralNetwork,  
        module__input_features=input_features,
        criterion=nn.BCEWithLogitsLoss,
        optimizer=optim.Adam,
        max_epochs=20,
        batch_size=64,
        verbose=0,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        iterator_train__shuffle=True,
        iterator_valid__shuffle=False,
    )
    

    param_space = {
        'module__hidden_size': Integer(64, 256),
        'module__num_layers': Integer(2, 4),
        'module__dropout_rate': Real(0.1, 0.6),
        'module__activation_function_name': Categorical(['relu', 'leaky_relu']),
        'lr': Real(1e-4, 1e-2, 'log-uniform'),
        'batch_size': Categorical([32, 64, 128]),
        'optimizer__weight_decay': Real(1e-6, 1e-3, 'log-uniform')
    }
    

    search = BayesSearchCV(
        estimator=net,
        search_spaces=param_space,
        n_iter=n_iter,
        cv=cv,
        scoring='f1',
        n_jobs=1,  
        random_state=42,
        verbose=1
    )
    
    print("Starting hyperparameter tuning...")
    search.fit(X_train, y_train)
    
    print(f"Best parameters: {search.best_params_}")
    print(f"Best cross-validation score: {search.best_score_:.4f}")
    
    return search.best_params_, search.best_score_, search

def main():

    train_df, test_df, val_df = load_data()
    

    X_train_processed, X_test_processed, X_val_processed, y_train, y_test, y_val, preprocessor = preprocess_data(
        train_df, test_df, val_df
    )
    
    print(f"Data shapes after preprocessing:")
    print(f"Train: {X_train_processed.shape}, Test: {X_test_processed.shape}, Val: {X_val_processed.shape}")
    

    print("\n=== Quick Training ===")
    input_features = X_train_processed.shape[1]
    model = NeuralNetwork(input_features)
    
    train_loader, test_loader, val_loader = create_data_loaders(
        X_train_processed, X_test_processed, X_val_processed, 
        y_train, y_test, y_val
    )
    
    train_losses, val_losses = train_model(model, train_loader, val_loader, epochs=10)
    test_accuracy = evaluate_model(model, test_loader)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    

    print("\n=== Hyperparameter Tuning ===")
    best_params, best_score, search = tune_classifier(X_train_processed, y_train, n_iter=10)
    
    return model, test_accuracy

if __name__ == "__main__":
    model, accuracy = main()
