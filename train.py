import argparse
import torch
from torchvision import datasets, transforms
from torch import nn
from torch import optim
import train_utils

def ParseCommandLine():
    parser = argparse.ArgumentParser()
    #-----Required Arguments----------
    parser.add_argument('data_directory', type=str, help='Directory of flower images')
    
    #-----Optional Arguments----------
    # Save directory => Set directory to save checkpoints
    parser.add_argument('--save_dir', type=str, help ='Directory to save checkpoints')
    #--------------------------------------------
    #arch => Choose architecture
    architectures = {'densenet121', 'densenet161','densenet201',
                     'vgg13', 'vgg16', 'vgg19'  }   
    parser.add_argument('--arch', dest = 'arch', default = 'vgg16', action='store',
                        choices=architectures, help='Pre-trained network to use')
    
    #--------------------------------------------
    #hyperparameters : Learning rate
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Model learning rate')
    parser.add_argument('--hidden_units', type=int, default = 512, help = 'Number of hidden layers in the model')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train')
    
    #--------------------------------------------
    #Use GPU for training
    parser.add_argument('--gpu', dest='gpu', action='store_true', default=False, help='Train with GPU')
    #parser.set_defaults(gpu=False)
    
    return parser.parse_args()


#----------------------------------------------------------------------------
def main(): 
    
    # Get Command Line Arguments
    args = ParseCommandLine()
    #Print data directory
    print("Data directory: ", args.data_directory)
    #Print device used
    use_gpu = torch.cuda.is_available() and args.gpu
    if use_gpu:
        print("Training on GPU.")
    else:
        print("Training on CPU.")
    
    #Print out architecture and hyperparameters
    print("Architecture: {}".format(args.arch))
    print("Learning rate: {}".format(args.learning_rate))
    print("Hidden units: {}".format(args.hidden_units))
    print("Epochs: {}".format(args.epochs))
    #Print our dave_dir option
    if args.save_dir:
        print("Checkpoint save directory: {}".format(args.save_dir))
    #--------------------------------------------------------------------
    # Get data loaders
    train_loader, valid_loader, test_loader, class_to_idx = train_utils.load_data(args.data_directory)
    #--------------------------------------------------------------------    
    # Build the model
    model = train_utils.build_model(args.arch, args.hidden_units)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    model.class_to_idx = class_to_idx
    #--------------------------------------------------------------------
    #Train the model
    train_utils.train_model(model, args.epochs, args.learning_rate, use_gpu,
                            criterion, optimizer, 
                            train_loader, valid_loader)
    #--------------------------------------------------------------------
    #Validation on the test set
    test_loss, accuracy = train_utils.validate_model(model, criterion, test_loader)
    print("Validation on the test set")
    print(f"Test accuracy: {accuracy:.2f}%")
    
    #--------------------------------------------------------------------
    # Save the checkpoint
    if input_args.save_dir:
        save_checkpoint(args.arch, args.learning_rate, args.hidden_units, args.epochs, 
                        model, optimizer,
                        args.save_directory)
        
if __name__ == "__main__":
    main()
    
