import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random

# implements the RRFS model with teacher-student architecture
# original code at https://github.com/alimirzaei/TSFS/blob/master/methods.py

# ---------------------------------------------------------
# MODULE 1: The Teacher (Autoencoder)
# ---------------------------------------------------------
class TeacherAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=10, seed=0):
        super(TeacherAutoencoder, self).__init__()

        # Encoder: Input -> Hidden 
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.enc_act = nn.Sigmoid()
        
        # Decoder: Hidden -> Input 
        self.regressor = nn.Linear(hidden_dim, 1)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # Returns both reconstruction and the latent representation
        latent = self.enc_act(self.encoder(x))
        regression = self.regressor(latent)
        reconstruction = self.decoder(latent)
        return reconstruction, regression, latent

    def get_representation(self, x):
        with torch.no_grad():
            return self.enc_act(self.encoder(x))

# ---------------------------------------------------------
# MODULE 2: The Student (Feature Selector)
# ---------------------------------------------------------
class StudentFeatureSelector(nn.Module):
    def __init__(self, input_dim, target_dim):
        super(StudentFeatureSelector, self).__init__()
        intermediate_dim = 10 * target_dim
        
        # Layer 1: The layer where we apply regularization for feature selection
        self.fc1 = nn.Linear(input_dim, intermediate_dim)
        self.act1 = nn.ReLU()
        
        # Layer 2: Maps to the teacher's latent space size
        self.fc2 = nn.Linear(intermediate_dim, target_dim)

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.fc2(x)
        return x

# ---------------------------------------------------------
# WRAPPER: Orchestrator
# ---------------------------------------------------------
class DeepFeatureSelection:
    def __init__(self, input_dim, hidden_dim=10, l1_reg=0.01, device='cpu'):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.l1_reg = l1_reg
        self.device = device
        
        # Initialize models
        self.teacher = TeacherAutoencoder(input_dim, hidden_dim).to(device)
        self.student = StudentFeatureSelector(input_dim, hidden_dim).to(device)

    def _l21_regularization(self, layer):
        """Calculates the L2,1 norm for group sparsity on the input layer."""
        # dim=0 is the output dimension, so taking norm over dim=0 collapses outputs
        # and leaves us with a vector of size [input_dim]
        group_norms = torch.norm(layer.weight, p=2, dim=0) 
        return self.l1_reg * torch.sum(group_norms)

    def train_teacher(self, X, y, X_test, y_test, epochs=10, batch_size=32, lr=0.001, recon_weight=1.0, reg_weight=1.0):
        """Trains the Autoencoder to learn the data representation."""
        print("Training Teacher (Autoencoder)...")
        optimizer = optim.Adam(self.teacher.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        # Prepare Data
        dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y.reshape(-1, 1)))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        

        train_losses = []
        test_losses = []
        self.teacher.train()
        for epoch in range(epochs):
            total_loss = 0
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad()
                recon, regression, _ = self.teacher(x_batch)
                loss = recon_weight * criterion(recon, x_batch) + reg_weight * criterion(regression, y_batch.view(-1, 1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            train_losses.append(total_loss / len(loader))
            test_loss = self.test_teacher(X_test, y_test, batch_size=batch_size, recon_weight=recon_weight, reg_weight=reg_weight)
            test_losses.append(test_loss)
            
            # Simple verbose output
            #if (epoch+1) % 5 == 0 or epoch == 0:
            #    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}, Test Loss: {test_loss:.4f}")

        return train_losses, test_losses
    
    def test_teacher(self, X, y, batch_size=32, recon_weight=1.0, reg_weight=1.0):
        """Evaluates the Autoencoder on test data."""
        #print("\nEvaluating Teacher (Autoencoder)...")
        criterion = nn.MSELoss()
        
        # Prepare Data
        dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y.reshape(-1, 1)))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        self.teacher.eval()
        total_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                recon, regression, _ = self.teacher(x_batch)
                loss = recon_weight * criterion(recon, x_batch) + reg_weight * criterion(regression, y_batch.view(-1, 1))
                total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        #print(f"Test Loss: {avg_loss:.4f}")
        return avg_loss

    def train_student(self, X, epochs=10, batch_size=32, lr=0.001):
        """
        Trains the Student to reconstruct the Teacher's representation 
        using row-sparsity regularization on the input layer.
        """
        print("\nTraining Student (Feature Selector)...")
        
        # 1. Get Teacher's learned representation (Targets for student)
        self.teacher.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        targets = self.teacher.get_representation(X_tensor).detach() # Detach to stop gradients to teacher
        
        # 2. Prepare Data (Input: X, Target: Teacher Latent Code)
        dataset = TensorDataset(X_tensor, targets)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = optim.Adam(self.student.parameters(), lr=lr)
        criterion = nn.MSELoss()

        train_losses = []
        test_losses = []
        
        self.student.train()
        for epoch in range(epochs):
            total_loss = 0
            mse_loss_accum = 0
            reg_loss_accum = 0
            
            for x_batch, y_batch in loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                preds = self.student(x_batch)
                
                # Calculate MSE
                mse_loss = criterion(preds, y_batch)
                
                # Calculate L2,1 Regularization on the first layer
                reg_loss = self._l21_regularization(self.student.fc1)
                
                # Total loss
                loss = mse_loss + reg_loss
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                mse_loss_accum += mse_loss.item()
                reg_loss_accum += reg_loss.item()

            train_losses.append(total_loss / len(loader))
            test_loss = self.test_student(X, epochs=epochs, batch_size=batch_size)
            test_losses.append(test_loss)
                
            #if (epoch+1) % 10 == 0 or epoch == 0:
            #    # print training and test loss
            #    print(f"Epoch {epoch+1}/{epochs}, Total Loss: {total_loss/len(loader):.4f}, Test Loss: {test_loss:.4f}")


        return train_losses, test_losses
                
            
                
    def test_student(self, X, epochs=10, batch_size=32):
        """Evaluates the Student on test data."""
        #print("\nEvaluating Student (Feature Selector)...")
        
        # Get Teacher's learned representation (Targets for student)
        self.teacher.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        targets = self.teacher.get_representation(X_tensor).detach() # Detach to stop gradients to teacher
        
        # Prepare Data
        dataset = TensorDataset(X_tensor, targets)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        criterion = nn.MSELoss()
        
        self.student.eval()
        total_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                
                preds = self.student(x_batch)
                
                mse_loss = criterion(preds, y_batch)
                total_loss += mse_loss.item()
        
        avg_loss = total_loss / len(loader)
        #print(f"Test MSE Loss: {avg_loss:.4f}")
        return avg_loss

    def get_feature_importance(self):
        """
        Calculates feature importance based on the weights of the Student's first layer.
        Matches: w = np.sum(np.square(w), 1) from original code.
        """
        self.student.eval()
        # PyTorch Linear weights are (out, in).
        # We want to sum squares across outputs for each input feature.
        weights = self.student.fc1.weight.detach().cpu().numpy()
        
        # Sum of squares across output dimension (axis 0 in PyTorch weight matrix)
        # Result corresponds to shape (input_dim,)
        importance = np.sum(np.square(weights), axis=0)
        return importance

