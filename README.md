# Inventory Monitoring at Distribution Centers

The goal of this project is to build a model that can count the number of objects in each bin. A system like this can be used to track inventory and make sure that delivery consignments have the correct number of items. The goal will be accomplish using the Amazon Bin Image Dataset.

## Project Set Up and Installation
Clone the project

```bash
  git clone https://github.com/JedSolo/license_plate_detection.git
```

Check the sagemaker.ipynb

Install dependencies

```terminal
  pip install torch
  
```

## Dataset

### Overview
The Amazon Bin Image Dataset contains over 500,000 images and metadata from bins of a pod in an operating Amazon Fulfillment Center. The bin images in this dataset are captured as robot units carry pods as part of normal Amazon Fulfillment Center operations.

### Access
AWS CLI Access (No AWS account required)
```bash
  aws s3 ls --no-sign-request s3://aft-vbi-pds/
```

## Model Training
1. Transfer learning - Using the resnet18 pretrained model
2. Hyperparameter: learning rate (0.001), batch size (64). These hyperparameter was choosen from a range of values through hyperparameter tuning. The steps are highlighted in the sagemaker.ipynb notebook.

## Machine Learning Pipeline
1. Data Preprocessing
It involved loading and transforming the image dataset to tensor and applying some data augmentation like resizing and random horizontal flip of the images

2. Feature Engineering
It involved normalizing the pixel values of the images

3. Model Training
Trained using a pretrained model (i.e. ResNet18)

4. Model Evaluation
The model was evaluated using accuracy, i.e the proportion of correctly classified examples over the whole set of examples. The accuracy for this model over the test set was 30%
