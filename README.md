# ERA-Assigments
Assignments submission 

<h1>About</h1>
<p>This assignment is related to ERA Session 5. Our task is to Move the contents of the code to the following files :
Move the contents of the code to the following files:<br>
  <strong>1.model.py</strong><br>
  <strong>2.utils.py</strong><br>
  <strong>3.S5.ipynb</strong>
  
Afte separating the codes, we then have to run the whole code again to check if the integration is working properly or not
  
<strong>We then have to Upload the code with the 4 files + README.md file to GitHub. README.md (look at the spelling) must have details about this code and how to read your code (what file does what). Heavy negative scores for not formatting your markdown file into p, H1, H2, list, etc.</strong>



</p>

<h1>How to use</h1>
<h3>Explanation of Utils.py</h3>

```import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from utils import *
 ```
<p>Above code imports all the libraries and frameworks for pytorch</p>

```# Train data transformations
train_transforms = transforms.Compose([
    transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
    transforms.Resize((28, 28)),
    transforms.RandomRotation((-15., 15.), fill=0),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    ])

# Test data transformations
test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])
```
<p>In above, we have defined the transformation for train and test dataset. We apply different kinds to transformation technique like random apply, center crop, rotation to train dataset only because we want to train out model for extreme conditions and cases also.<br>
But for test dataset, we are only using simple tensor and normalize
</p>

```def train(model, device, train_loader, optimizer, criterion):
  model.train()
  pbar = tqdm(train_loader)

  train_loss = 0
  correct = 0
  processed = 0

  for batch_idx, (data, target) in enumerate(pbar):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()

    # Predict
    pred = model(data)

    # Calculate loss
    loss = criterion(pred, target)
    train_loss+=loss.item()

    # Backpropagation
    loss.backward()
    optimizer.step()
    
    correct += GetCorrectPredCount(pred, target)
    processed += len(data)

    pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

  train_acc.append(100*correct/processed)
  train_losses.append(train_loss/len(train_loader))

def test(model, device, test_loader, criterion):
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += criterion(output, target, reduction='sum').item()  # sum up batch loss

            correct += GetCorrectPredCount(output, target)


    test_loss /= len(test_loader.dataset)
    test_acc.append(100. * correct / len(test_loader.dataset))
    test_losses.append(test_loss)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
```

<p>In above, we have define the train and test function. In train function, we are loading train loader data, finding loss, doing backward function but for test function, w don't do backward function simply because we don't want to calculate backpropagation for test dataset</p>
