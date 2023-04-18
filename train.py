#Importing dependencies.
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets

import argparse
import os

#TODO: Import dependencies for Debugging andd Profiling
try:
    import smdebug
    from smdebug import modes
    from smdebug.profiler.utils import str2bool
    from smdebug.pytorch import get_hook
    
except ModuleNotFoundError:
    print("module 'smdebug' is not installed. Probably an inference container")
    


    
def test(model, test_loader, criterion, hook):
    model.eval()
    hook.set_mode(modes.PREDICT) # Set the SMDebug hook for the testing phase
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
        print(f"\nTest Loss: {test_loss/len(test_loader.dataset)}, Accuracy: {100*(correct/len(test_loader.dataset))}%\n")
        

def train(model, train_loader, validation_loader, criterion, optimizer, hook):
    best_loss = 1e6
    epochs=50
    image_dataset = {'train': train_loader, 'valid': validation_loader}
    loss_counter = 0
    
    for epoch in range(epochs):
        for phase in ['train', 'valid']:
            print(f"Epoch {epoch}, Phase {phase}")
            if phase == "train":
                model.train()
                hook.set_mode(modes.TRAIN)  # Set the SMDebug hook for the training phase
                
            else:
                model.eval()
                hook.set_mode(modes.EVAL) # Set the SMDebug hook for the validation phase.
                
            running_loss = 0.0
            correct = 0.0
            running_samples = 0
            
            for step, (data, target) in enumerate(image_dataset[phase]):
                outputs = model(data)
                loss = criterion(outputs, target)
                
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * data.size(0)
                correct += torch.sum(preds == target.data).item()
                running_samples += len(data)
                if running_samples % 2000  == 0:
                    accuracy = correct/running_samples
                    print("Images [{}/{} ({:.0f}%)] Loss: {:.2f} Accuracy: {}/{} ({:.2f}%)".format(
                                running_samples,
                                len(image_dataset[phase].dataset),
                                100.0 * (running_samples / len(image_dataset[phase].dataset)),
                                loss.item(),
                                correct,
                                running_samples,
                                100.0*accuracy,
                            )
                        )
            
            epoch_loss = running_loss / running_samples
            epoch_acc = correct / running_samples
            
            if phase == 'valid':
                if epoch_loss<best_loss:
                    best_loss=epoch_loss
                else:
                    loss_counter+=1 
        
        if loss_counter == 1:
            break
            
    return model

    
def net():
    model = models.resnet18(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
        
    num_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_features, 133))
    
    return model

def create_data_loaders(data, batch_size, test_batch_size):
    train_data_path = os.path.join(data, 'train')
    test_data_path = os.path.join(data, 'test')
    validation_data_path=os.path.join(data, 'valid')
        
    # define data transforms
    train_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        
    test_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            
    # load data
    train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=test_transform)
    test_loader  = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size, shuffle=False)

    validation_data = torchvision.datasets.ImageFolder(root=validation_data_path, transform=test_transform)
    validation_loader  = torch.utils.data.DataLoader(validation_data, batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader, validation_loader
    
    
def main(args):
    # Initialize model
    model = net()
    
    train_loader, test_loader, validation_loader=create_data_loaders(args.data, args.batch_size, args.test_batch_size)
    
    # Define loss function and optimizer
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # create debugger hook
    hook = get_hook(create_if_not_exists=True)
    
    if hook:
        hook.register_hook(model)
        hook.register_loss(loss_criterion)
        
    # Train the model
    model = train(model, train_loader, validation_loader, loss_criterion, optimizer, hook)

    # Test the model
    test(model, test_loader, loss_criterion, hook)

    # save the model
    path = os.path.join(args.model_dir, 'model.pth')
    torch.save(model.state_dict(), path)
    

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--test-batch-size', type=int, default=100)
    parser.add_argument('--data', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--model-dir', type=str, default=os.environ["SM_MODEL_DIR"])
    
    args=parser.parse_args()
    
    main(args)