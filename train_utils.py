import torch
from torch import nn
#from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict



def load_data(data_dir):
    """
    Load data from data_dir
    Define transforms for the training, validation, and testing sets
    Load the datasets with ImageFolder
    Using the image datasets and the trainforms, define the dataloaders
    => Return dataloaders for training, validation and testing datasets.
    """
    train_dir = data_dir + '/train/'
    valid_dir = data_dir + '/valid/'
    test_dir = data_dir + '/test/'
    #----------------------------------------------------------------
    #Define transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])]) 
    #Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(valid_dir, transform=test_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    #Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)
    
    class_to_idx = train_data.class_to_idx
    return train_loader, valid_loader, test_loader, class_to_idx


def build_model(arch, hidden_layers):
    '''
    Inputs : chosen architecture ('densenet121', 'densenet161','densenet201',
                                  'vgg13', 'vgg16', 'vgg19'
             hidden_units
    Load pretrained model
    Define a new, untrained feed-forward network as a classifier
    => Return untrained model
    '''
    print("################# BUILD MODEL ###############################")
    if arch == 'vgg13':
        model = models.vgg13_bn(pretrained=True)
        classifier_input_size = model.classifier[0].in_features
    elif arch == 'vgg16':
        model = models.vgg16_bn(pretrained=True)
        classifier_input_size = model.classifier[0].in_features
    elif arch == 'vgg19':
        model = models.vgg19_bn(pretrained=True)
        classifier_input_size = model.classifier[0].in_features
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        classifier_input_size = model.classifier.in_features
    elif arch == 'densenet161':
        model = models.densenet161(pretrained=True)
        classifier_input_size = model.classifier.in_features
    elif arch == 'densenet201':
        model = models.densenet201(pretrained=True)
        classifier_input_size = model.classifier.in_features
    else:
        raise Exception("Unknown model")

    #print("Input size: ", classifier_input_size)
    output_size = 102
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
        
    classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(classifier_input_size, hidden_layers)),
                            ('relu', nn.ReLU()),
                            ('dropout', nn.Dropout(0.2)),
                            ('fc2', nn.Linear(hidden_layers, output_size)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))

    model.classifier = classifier    

    return model

#-----------------------------------------------------------------
def train_model (model, epochs, learning_rate, use_gpu, criterion, optimizer, training_loader, validation_loader):
    """
    Train the classifier layers using backpropagation using the pre-trained network to get the features
    """
    print("################# TRAIN MODEL ###############################")
    model.train()
    print_every = 10
    steps = 0
    #validation_losses, train_losses = [], []

    #Determine which device shall be used depending on availibility of GPU
    if use_gpu == True:
        device = "cuda"
    else:
        device = "cpu"   
    #print("Used device = ", device)
    model.to(device);

    for epoch in range(epochs):
        running_loss = 0
        for inputs, labels in iter(training_loader):
            steps += 1
            # Move input and label tensors to the used device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                validation_loss, accuracy = validate_model(model, criterion, validation_loader)

                #train_losses.append(running_loss/print_every)
                #validation_losses.append(validation_loss)

                print("Epoch: {}/{} ".format(epoch+1, epochs),
                      "Training Loss: {:.3f} ".format(running_loss/print_every),
                      "Validation Loss: {:.3f} ".format(validation_loss),
                      "Validation Accuracy: {:.2f}%".format(accuracy))
                running_loss = 0

                # Put model back in training mode
                model.train()
    #return train_losses, validation_losses
    
#-----------------------------------------------------------------    
def validate_model(model, criterion, data_loader):
    model.eval()
    accuracy = 0
    test_loss = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():    
      for inputs, labels in iter(data_loader):
          # Move input and label tensors to the default device
          inputs, labels = inputs.to(device), labels.to(device)

          output = model.forward(inputs)
          test_loss += criterion(output, labels).item()
          ps = torch.exp(output).data 
          equality = (labels.data == ps.max(1)[1])
          accuracy += equality.type_as(torch.FloatTensor()).mean()

    return test_loss/len(data_loader), (accuracy/len(data_loader))*100 


#-----------------------------------------------------------------
def save_checkpoint(arch, learning_rate, hidden_units, epochs, 
                    model, optimizer,
                    save_directory):
    ''' 
    Save the checkpoint
    '''
    print("################# SAVE CHECKPOINT ###############################")
    state = {
        'arch': arch,
        'learning_rate': learning_rate,
        'hidden_units': hidden_units,
        'epochs': epochs,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'class_to_idx': model.class_to_idx
    }

    torch.save(state, save_directory)
    
