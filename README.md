# Convolutional Autoencoder for Image Denoising

## AIM

To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset


## DESIGN STEPS

### STEP 1:
Import Required Libraries.

### STEP 2:
Define Data Transformation and Load MNIST Dataset.

### STEP 3:
Add Noise to Images and Build the Denoising Autoencoder Model.
Write your own steps
### STEP 4:
Initialize Model, Loss Function, and Optimizer and train the model.
### STEP 5:
Finally test the model and get the output.

## PROGRAM
### Name: YASEEN F
### Register Number:212223220126


```
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
      super(DenoisingAutoencoder,self).__init__()
      self.encoder=nn.Sequential(
          nn.Conv2d(1, 16, 3, stride=2, padding=1),
          nn.ReLU(),
          nn.Conv2d(16, 32, 3, stride=2, padding=1),
          nn.ReLU()
      )
      self.decoder=nn.Sequential(
          nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
          nn.ReLU(),
          nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
          nn.Sigmoid()
      )
    def forward(self,x):
      x=self.encoder(x)
      x=self.decoder(x)
      return x
```
```
model = DenoisingAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)
```
```
def train(model, loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for data in loader:
            inputs, _ = data
            inputs = inputs.to(device)
            noisy_inputs = add_noise(inputs)
            noisy_inputs = noisy_inputs.to(device)
            optimizer.zero_grad()
            outputs = model(noisy_inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(loader.dataset)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}')
```

## OUTPUT

### Model Summary

<img width="790" height="500" alt="image" src="https://github.com/user-attachments/assets/a8efeb0b-ef6b-441c-830d-330064c844c3" />

### Original vs Noisy Vs Reconstructed Image

<img width="1738" height="717" alt="image" src="https://github.com/user-attachments/assets/42b69300-f9fa-4923-a660-dcdea81b8e8d" />



## RESULT
The Denoising Autoencoder was implemented and trained using the MNIST dataset. The model successfully removed noise from images and reconstructed clearer handwritten digits.
