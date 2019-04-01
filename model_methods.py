#PROGRAMMER: BESHER TABBARA
#DATE: 3/31/2019

import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import copy
import time
import json

def load_model(arch='vgg16', num_outputs=102, hidden_units=4096):
    '''
        Builds the requested architecture with either a target classifier
        or modified layers to match output classes depending on neural
        network architecture
    '''
        
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        num_in_features = model.classifier[0].in_features
        rebuild_classifier = True
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)
        num_in_features = model.classifier[1].in_features
        rebuild_classifier = True
    elif arch == 'densenet':
        model = models.densenet161(pretrained=True)
        num_in_features = model.classifier.in_features
        rebuild_classifier = True
    elif arch == 'inception':
        model = models.inception_v3(pretrained=True)
        rebuild_classifier = False
    elif arch == 'resnet':
        model = models.resnet18(pretrained=True)
        rebuild_classifier = False
    else:
        raise ValueError('Architecture Not Available', arch)
        
    #freeze the feature extraction parameters of the main model
    for param in model.parameters():
        param.requires_grad = False
        
    if rebuild_classifier:
        #rebuilding the classifier layer for VGG16, AlexNet and DenseNet
        model.classifier = nn.Sequential(nn.Linear(num_in_features, hidden_units),
                                         nn.ReLU(),
                                         nn.Dropout(0.2),
                                         nn.Linear(hidden_units, int(hidden_units/4)),
                                         nn.ReLU(),
                                         nn.Dropout(0.2),
                                         nn.Linear(int(hidden_units/4), num_outputs),
                                         nn.LogSoftmax(dim=1))
    elif arch == 'inception':
        #reshape both layers to finetune inception v3 
        #handle the auxilary net
        num_features = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(num_features, num_outputs)
        #handle the primary net
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_outputs) 
    elif arch == 'resnet':
        #modify the fc layer to match number of classes
        model.fc = nn.Linear(512, num_outputs)
       
    return model
    
def train_model(image_dataloaders, arch='vgg16', num_outputs=102, hidden_units=4096, 
                learning_rate=0.001, epochs=20, gpu=False, checkpoint=''):
    '''
        Trains the model on given dataloaders
    '''
    #load the model
    model = load_model(arch, num_outputs, hidden_units)
    #attach ancillary information about the nn to the model object
    model.arch = arch
    model.num_outputs = num_outputs
    model.hidden_units = hidden_units
    model.learning_rate = learning_rate
    model.epochs = epochs
    model.gpu = gpu
    model.checkpoint = checkpoint
    
    print('Architecture: ',arch,'Hidden units: ',hidden_units)
    print('Training epochs: ',epochs, 'Learning rate: ',learning_rate)
    print('Trianing data size: {} images, '.format(len(image_dataloaders['train'].dataset)),
          'validation data size: {} images'.format(len(image_dataloaders['valid'].dataset)))
     
    #use gpu if selected and available
    if gpu and torch.cuda.is_available():
        print('On GPU')
        device = torch.device("cuda:0")
        model.cuda()
    else:
        print('On CPU')
        device = torch.device("cpu")     
        
    #setup the loss function
    if arch == 'inception' or 'resnet':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.NLLLoss()
    
    #only the new or modified layers will get gradient updates
    print("Parameters to learn:")
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
    #making sure only added parameters are being optimized with 
    #gradient adjustments during training
    optimizer = optim.Adam(params_to_update, lr=learning_rate)

    #resetting accuracy and deep copying the model weights/biases
    best_accuracy = 0
    best_model_weights = copy.deepcopy(model.state_dict())
    
    #to keep track of the losses throughout training
    train_losses, valid_losses = [], []
    
    print_every = 100 #for debugging
    start_time = time.time() 

    for e in range(epochs):
        epoch_accuracy = 0 
        running_loss = 0
        steps = 0
        start_training_time_per_steps = time.time()
            
        for images, labels in image_dataloaders['train']:
            images, labels = images.to(device), labels.to(device)
            steps += 1
            optimizer.zero_grad()
            
            #run training data through the model
            if arch == 'inception':
                #From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                outputs, aux_outputs = model(images)
                loss1 = criterion(outputs, labels)
                loss2 = criterion(aux_outputs, labels)
                loss = loss1 + 0.4 * loss2
            else:
                output_logps = model(images)
                loss = criterion(output_logps, labels)
            
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            
            #perform validation at "print_every"
            if steps % print_every == 0:
                #calculate the training time per steps
                training_time_per_steps = time.time() - start_training_time_per_steps
                #reset the accuracy and validation loss
                accuracy, valid_loss = 0, 0
                #put the model in evaluation mode for quicker validation
                model.eval()
                #we're not doing any gradient related calculations when punching 
                #through the validation data
                with torch.no_grad():
                    valid_start_time = time.time()
                    for images, labels in image_dataloaders['valid']:
                        images, labels = images.to(device), labels.to(device)
                        valid_logps = model(images)
                        #calculate the validation loss before taking the exp
                        valid_loss += criterion(valid_logps, labels)
                        
                        valid_ps = torch.exp(valid_logps)
                        top_p, top_class = valid_ps.topk(1, dim=1)
                        equality = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equality.type(torch.FloatTensor)).item()
                    
                valid_time = time.time() - valid_start_time
                #keeping track of the losses to plot later in case we need to
                train_losses.append(running_loss/steps)
                valid_losses.append(valid_loss/len(image_dataloaders['valid']))
                epoch_accuracy = accuracy/len(image_dataloaders['valid'])
                    
                #printing losses, accuracy, etc. as we train
                print('Epoch {}/{} '.format(e+1, epochs),
                      'Step {} '.format(steps),
                      'Train loss: {:.3f} '.format(running_loss/steps),
                      'Valid loss: {:.3f} '.format(valid_loss/len(image_dataloaders['valid'])),
                      'Accuracy: {:.2f}% '.format(epoch_accuracy*100),
                      'Train dur: {:.1f}s '.format(training_time_per_steps),
                      'Valid dur: {:.1f}s'.format(valid_time))
                #reset the running loss to zero and put the model back into training mode
                running_loss = 0
                model.train()
                start_training_time_per_steps = time.time()
        
        #saving the best weights and biases based on best accuracy   
        if(epoch_accuracy > best_accuracy):
            best_accuracy = epoch_accuracy
            best_model_wts = copy.deepcopy(model.state_dict())
    
    #loading model object with best weights
    model.load_state_dict(best_model_wts)
    
    #storing dir_to_cat into the model object - added this for easier lookup
    with open('dir_to_cat.json', 'r') as f:
        dir_to_cat = json.load(f)
    
    #saving train and valid losses to the model in case we need to access them
    model.train_losses = train_losses
    model.valid_losses = valid_losses
    
    #printing total training time and best accuracy
    total_time = time.time() - start_time
    print('Time for complete training {:.0f}m {:.0f}s'.format(total_time // 60, total_time % 60))
    print('Best accuracy: {:3f}%'.format(best_accuracy*100))

    #saving checkpoint if requested
    if checkpoint:
        print ('Checkpoint saved to:', checkpoint) 
        checkpoint_dict = {'arch': arch,
                           'dir_to_cat': dir_to_cat,
                           'hidden_units': hidden_units,
                           'best_accuracy': best_accuracy,
                           'best_model_weights': best_model_wts,
                           'train_losses': train_losses,
                           'valid_losses': valid_losses}
        torch.save(checkpoint_dict, checkpoint)
    
    #return the model object with best weights and biases
    return model