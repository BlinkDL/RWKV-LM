import torch
import torch.nn as nn
import torch.jit
import time

# Define a more complex model
    
class ComplexModel(nn.Module):
    def __init__(self):
        super(ComplexModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate the model
model = ComplexModel()

# Create a large input tensor (batch of images, e.g. 128 images of size 32x32 with 3 channels)
input_tensor = torch.randn(128, 3, 32, 32)

# Test execution time without @torch.jit.script
start = time.time()
for _ in range(100):
    output = model(input_tensor)
end = time.time()
print(f"Execution time without @torch.jit.script: {end - start:.6f} seconds")

# Now script the model
scripted_model = torch.jit.script(model)

# Test execution time with @torch.jit.script
start = time.time()
for _ in range(100):
    output = scripted_model(input_tensor)
end = time.time()
print(f"Execution time with @torch.jit.script: {end - start:.6f} seconds")

