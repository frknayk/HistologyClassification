import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import wandb

from histologyai.dataset_loader import InferenceDataset
from histologyai.utils import convert_path
from pathlib import Path

def log_inference_model(model, test_loeader, device):
    model.eval()
    with torch.no_grad():
        for i, image in enumerate(test_loader):
            output = model(image.to(device))
            probabilities = torch.softmax(output, dim=1)
            _, predicted = torch.max(output, 1)
            # Bring the image back to the CPU for wandb logging
            image = image.cpu()
            # Log the predicted class and probabilities to wandb
            wandb.log({f"Test Image {i + 1} Prediction": int(predicted)})
            wandb.log({f"Test Image {i + 1} Probabilities": [float(p) for p in probabilities[0]]})
            # Log the image along with predictions as an artifact
            image_artifact = wandb.Image(image.squeeze())
            wandb.log({f"Test Image {i + 1}": image_artifact})


if __name__ == "__main__":    
    # Load the trained model checkpoint
    model_checkpoint = convert_path('logs\\checkpoints\\ResNet50Classifier_ADAM_128batch_20230902_192503\\best_0.9062.pt')
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
    wandb.init(project="HistologyInference",dir="logs/inference/")
    # Perform predictions and log images with predictions to wandb
    log_inference_model(model, test_loader, device)

    # Finish the wandb run
    wandb.finish()
