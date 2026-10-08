import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from customDataset import ImageDataset
from pathlib import Path


def save_checkpoint(state, filename='Model/training_checkpoint.pht.tar'):
    print('=> Saving checkpoint')
    torch.save(state, filename)

def load_checkpoint(checkpoint):
    print('=> Loading checkpoint')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


#Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



#Parameters
input_size = 1024
num_classes = 2
batch_size = 64
learning_rate = 0.001
num_epochs = 50
load_model = False


#Load Data
current_folder = Path(__file__).resolve().parent.parent

training_data = ImageDataset(csv_file=current_folder / 'Images/training.csv', 
                            img_dir=current_folder / 'Images',
                            transform= transforms.ToTensor()
                            )

validation_data = ImageDataset(csv_file=current_folder / 'Images/validate.csv', 
                            img_dir=current_folder / 'Images',
                            transform= transforms.ToTensor()
                            )

test_data = ImageDataset(csv_file=current_folder / 'Images/test.csv', 
                            img_dir=current_folder / 'Images',
                            transform= transforms.ToTensor()
                            )

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=False)
validation_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


# Load MobileNetV3-Small pretrained on ImageNet
model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)

# Modify the final layer for a custom number of classes (e.g., 10)
model.classifier[3] = nn.Linear(in_features=input_size, out_features=2)

model.to(device=device)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

if load_model:
    load_checkpoint(torch.load('Model/training_checkpoint.pht.tar'))

# Training loop
model.train()
for epoch in range(num_epochs):
    losses = []

    if epoch % 3 == 0:
        checkpoint = {'state_dict' : model.state_dict(), 'optimizer' : optimizer.state_dict()}
        save_checkpoint(checkpoint)

    for batch_idx, (data, targets) in enumerate(train_dataloader):
        #Get data to cuda
        data = data.to(device=device)
        targets = targets.to(device=device)

        #forward
        scores = model(data)
        loss = criterion(scores, targets)

        losses.append(loss.item())

        #backward
        optimizer.zero_grad()
        loss.backward()

        #gradient descent / adam step
        optimizer.step()

    print(f"Cost at epoch {epoch} is {sum(losses)/len(losses):.5f}")



def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with accuaracy {float(num_correct)/(num_samples)*100:.2f}')

    model.train()

    return float(num_correct)/(num_samples)*100

print('Checking accuracy on Training Set')
check_accuracy(train_dataloader, model)

print('Checking accuracy on Test Set')
accuracy = check_accuracy(test_dataloader, model)

torch.save({
    'state_dict': model.state_dict(),
    'optimizer' : optimizer.state_dict(),
    'loss' : sum(losses)/len(losses),
    'accuracy': accuracy
}, 'Model/trueModel.pht.tar')