import matplotlib.pyplot as plt
import torch
import torchvision

from torch import nn
from torchvision import transforms
from helper_functions import set_seeds

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device=",device)

# 1. Get pretrained weights for ViT-Base
pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT

# 2. Setup a ViT model instance with pretrained weights
pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights).to(device)
# pretrained_vit = torchvision.models.vit_b_16(weights=None).to(device)

# checkpoint_filename = 'E:/Deep learning/Project/code_data/Image-Classification-Using-Vision-transformer-main/weights/' + 'pre_trained.pt'
# torch.save(pretrained_vit.state_dict(), checkpoint_filename)
# exit()

# 3. Freeze the base parameters
for parameter in pretrained_vit.parameters():
    parameter.requires_grad = False

# 4. Change the classifier head
# class_names = ['apple_pie', 'baby_back_ribs',
#                'baklava',
#                'beef_carpaccio',
#                'beef_tartare',
#                'beet_salad',
#                'beignets',
#                'bibimbap',
#                'bread_pudding',
#                'breakfast_burrito',
#                'bruschetta',
#                'caesar_salad',
#                'cannoli',
#                'caprese_salad',
#                'carrot_cake',
#                'ceviche',
#                'cheesecake',
#                'cheese_plate',
#                'chicken_curry',
#                'chicken_quesadilla',
#                'chicken_wings',
#                'chocolate_cake',
#                'chocolate_mousse',
#                'churros',
#                'clam_chowder',
#                'club_sandwich',
#                'crab_cakes',
#                'creme_brulee',
#                'croque_madame',
#                'cup_cakes',
#                'deviled_eggs',
#                'donuts',
#                'dumplings',
#                'edamame',
#                'eggs_benedict',
#                'escargots',
#                'falafel',
#                'filet_mignon',
#                'fish_and_chips',
#                'foie_gras',
#                'french_fries',
#                'french_onion_soup',
#                'french_toast',
#                'fried_calamari',
#                'fried_rice',
#                'frozen_yogurt',
#                'garlic_bread',
#                'gnocchi',
#                'greek_salad',
#                'grilled_cheese_sandwich',
#                'grilled_salmon',
#                'guacamole',
#                'gyoza',
#                'hamburger',
#                'hot_and_sour_soup',
#                'hot_dog',
#                'huevos_rancheros',
#                'hummus',
#                'ice_cream',
#                'lasagna',
#                'lobster_bisque',
#                'lobster_roll_sandwich',
#                'macaroni_and_cheese',
#                'macarons',
#                'miso_soup',
#                'mussels',
#                'nachos',
#                'omelette',
#                'onion_rings',
#                'oysters',
#                'pad_thai',
#                'paella',
#                'pancakes',
#                'panna_cotta',
#                'peking_duck',
#                'pho',
#                'pizza',
#                'pork_chop',
#                'poutine',
#                'prime_rib',
#                'pulled_pork_sandwich',
#                'ramen',
#                'ravioli',
#                'red_velvet_cake',
#                'risotto',
#                'samosa',
#                'sashimi',
#                'scallops',
#                'seaweed_salad',
#                'shrimp_and_grits',
#                'spaghetti_bolognese',
#                'spaghetti_carbonara',
#                'spring_rolls',
#                'steak',
#                'strawberry_shortcake',
#                'sushi',
#                'tacos',
#                'takoyaki',
#                'tiramisu',
#                'tuna_tartare',
#                'waffles']


class_names = ['garlic_bread','hot_dog','ice_cream','omelette','pizza']

set_seeds()
pretrained_vit.heads = nn.Linear(in_features=768, out_features=len(class_names)).to(device)
# pretrained_vit # uncomment for model output

from torchinfo import summary

# Print a summary using torchinfo (uncomment for actual output)
summary(model=pretrained_vit,
        input_size=(32, 3, 224, 224), # (batch_size, color_channels, height, width)
        # col_names=["input_size"], # uncomment for smaller output
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
)

train_dir='E:/Deep learning/Project/OTHER_DATA/2/archive/food-101/food-101/sub_v2/subset_train_v2/'
test_dir='E:/Deep learning/Project/OTHER_DATA/2/archive/food-101/food-101/sub_v2/subset_test_v2/'

# Get automatic transforms from pretrained ViT weights
pretrained_vit_transforms = pretrained_vit_weights.transforms()
print(pretrained_vit_transforms)


import os
NUM_WORKERS = 0
print(NUM_WORKERS)

import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# NUM_WORKERS = os.cpu_count()-2
NUM_WORKERS=0

def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int=NUM_WORKERS
):

  # Use ImageFolder to create dataset(s)
  train_data = datasets.ImageFolder(train_dir, transform=transform)
  test_data = datasets.ImageFolder(test_dir, transform=transform)

  # Get class names
  class_names = train_data.classes

  # Turn images into data loaders
  train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=False,
  )
  test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      pin_memory=False,
  )

  return train_dataloader, test_dataloader, class_names


# Setup dataloaders
train_dataloader_pretrained, test_dataloader_pretrained, class_names = create_dataloaders(train_dir=train_dir,
                                                                                                     test_dir=test_dir,
                                                                                                     transform=pretrained_vit_transforms,
                                                                                                     batch_size=6) # Could increase if we had more samples, such as here: https://arxiv.org/abs/2205.01580 (there are other improvements there too...)


from going_modular.going_modular import engine

from torch.optim.lr_scheduler import CosineAnnealingLR

# Create optimizer and loss function
optimizer = torch.optim.Adam(params=pretrained_vit.parameters(),
                             lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()
scheduler = CosineAnnealingLR(optimizer, T_max=10)


# Train the classifier head of the pretrained ViT feature extractor model
set_seeds()


results = engine.train(model=pretrained_vit,
                       train_dataloader=train_dataloader_pretrained,
                       test_dataloader=test_dataloader_pretrained,
                       optimizer=optimizer,
                       scheduler = scheduler,
                       loss_fn=loss_fn,
                       epochs=2,
                       device=device)


# Plot the loss curves
from helper_functions import plot_loss_curves

plot_loss_curves(results)