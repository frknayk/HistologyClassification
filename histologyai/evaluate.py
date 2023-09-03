import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import wandb

from histologyai.dataset_loader import InferenceDataset
from histologyai.utils import convert_path
from pathlib import Path

def log_inference_result(model, test_loader, device, classes):
    """Run inference on trained model and unlabelled data, log results as wand table"""
    # assume a model has returned predictions on four images
    # with the following fields available:
    # - the image id
    # - the image pixels, wrapped in a wandb.Image()
    # - the model's predicted label
    # my_data = [
    #   [0, wandb.Image("img_0.jpg"), 0],
    #   [1, wandb.Image("img_1.jpg"), 8],
    #   [2, wandb.Image("img_2.jpg"), 7],
    #   [3, wandb.Image("img_3.jpg"), 1]
    # ]
    my_data = []
    model.eval()
    with torch.no_grad():
        for i, image in enumerate(test_loader):
            output = model(image.to(device))
            probabilities = torch.softmax(output, dim=1)
            _, predicted = torch.max(output, 1)
            # Bring the image back to the CPU for wandb logging
            image = image.cpu()
            new_data = [i, wandb.Image(image.squeeze()), classes[predicted]]
            my_data.append(new_data)
    # create a wandb.Table() with corresponding columns
    columns=["id", "image", "prediction"]
    test_table = wandb.Table(data=my_data, columns=columns)
    wandb.log({"Inference Results":test_table})

def log_test_result(model, test_loader, device, classes):
    """Run inference on trained model and labelled data, log results as wand table"""
    # assume a model has returned predictions on four images
    # with the following fields available:
    # - the image id
    # - the image pixels, wrapped in a wandb.Image()
    # - the model's predicted label
    # - the ground truth label
    # my_data = [
    #   [0, wandb.Image("img_0.jpg"), 0, 1],
    #   [1, wandb.Image("img_1.jpg"), 8, 6],
    #   [2, wandb.Image("img_2.jpg"), 7, 5],
    #   [3, wandb.Image("img_3.jpg"), 1, 3]
    # ]
    my_data = []
    model.eval()
    with torch.no_grad():
        for i, (image,label) in enumerate(test_loader):
            output = model(image.to(device))
            probabilities = torch.softmax(output, dim=1)
            _, predicted = torch.max(output, 1)
            # Bring the image back to the CPU for wandb logging
            image = image.cpu()
            new_data = [i, wandb.Image(image.squeeze()), classes[predicted], classes[image.target]]
            my_data.append(new_data)
    # create a wandb.Table() with corresponding columns
    columns=["id", "image", "prediction", "ground_truth"]
    test_table = wandb.Table(data=my_data, columns=columns)
    wandb.log({"Test Results":test_table})


if __name__ == "__main__":    
    # Load the trained model checkpoint
    model_checkpoint = convert_path('logs\\checkpoints\\DENEME_resnet50_Histology_ADAM_Batch128_20230903_145241\\best_0.2812.pt')
    from histologyai.models.ResNet50Classifier import ResNet50Classifier
    model = ResNet50Classifier(4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(torch.load(model_checkpoint)["model_state_dict"])
    # Define data transformations for the test set
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Adjust size as needed
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Modify normalization values
    ])
    # Create a DataLoader for the test set
    test_data = InferenceDataset('data/Histology_Dataset/Test', transform=test_transform)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    # Initialize wandb
    wandb.init(project="HistologyInference",dir="logs_inference_deneme/")
    # Perform predictions and log images with predictions to wandb
    log_inference_result(model, test_loader, device)
    # Finish the wandb run
    wandb.finish()
