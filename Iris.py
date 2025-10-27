import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

iris = load_iris()
X = iris.data
y = iris.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)


print(f"Eğitim Verisi Boyutu: {X_train_tensor.shape}") # (120, 4)
print(f"Test Verisi Boyutu: {X_test_tensor.shape}")   # (30, 4)


INPUT_SIZE = X_train.shape[1]
HIDDEN_SIZE = 10
NUM_CLASSES = len(np.unique(y))

class IrisClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(IrisClassifier, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)

        self.fc2 = nn.Linear(hidden_size, num_classes)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = IrisClassifier(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES)
print("Model Yapısı:\n", model)

LEARNING_RATE = 0.2
NUM_EPOCHS = 10

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(NUM_EPOCHS):

    optimizer.zero_grad()

    outputs = model(X_train_tensor)
    

    loss = criterion(outputs, y_train_tensor)
    

    loss.backward()

    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Kayıp (Loss): {loss.item():.4f}')

print("\nEğitim Tamamlandı.")

model.eval() 

with torch.no_grad():
    
    test_outputs = model(X_test_tensor)
    
    _, predicted = torch.max(test_outputs.data, 1) 
    
    total = y_test_tensor.size(0)
    correct = (predicted == y_test_tensor).sum().item()
    
    accuracy = 100 * correct / total

print(f"\nTest Verisi Üzerindeki Toplam Doğruluk: {accuracy:.2f}%")