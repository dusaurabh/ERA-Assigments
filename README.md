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
