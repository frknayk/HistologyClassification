import os
import time
import torch
import glob

    
def convert_path(path):
    if os.name == 'nt':  # Windows
        return path.replace('/', '\\')
    else:  # Linux
        return path.replace('\\', '/')


class ExperimentManager:
    def __init__(self, exp_name):
        self.experiment_name = self._generate_experiment_name(exp_name)
        self.experiment_dir = os.path.join("logs", "checkpoints", self.experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)

    def _generate_experiment_name(self, exp_name):
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        return f"{exp_name}_{timestamp}"

    def save_checkpoint(self, model, optimizer, epoch, loss, accuracy):
        checkpoint_filename = f"epoch_{epoch}.pt"
        checkpoint_path = os.path.join(self.experiment_dir, checkpoint_filename)
        save_checkpoint(model, optimizer, epoch, loss, accuracy, checkpoint_path)
        
def save_checkpoint(model, optimizer, epoch, loss, accuracy, filename, num_best_models=5):
    """
    Save model checkpoint to a file based on a certain frequency or if a better accuracy is achieved.

    Args:
        model (torch.nn.Module): The PyTorch model to be saved.
        optimizer (torch.optim.Optimizer): The optimizer's state to be saved.
        epoch (int): The current training epoch.
        loss (float): The current training loss.
        val_accuracy (float): The current validation accuracy.
        filename (str): The path to the checkpoint file.
        num_best_models (int): The number of best models to save.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'training_loss': loss,
        'val_accuracy': accuracy,
    }
    # Track the best models based on accuracy and save up to num_best_models
    checkpoint_dir = os.path.dirname(filename)
    files_in_dir = glob.glob(os.path.join(checkpoint_dir, 'best_*.pt'))
    best_models = sorted(files_in_dir, key=lambda x: float('.'.join(\
        os.path.basename(x).split("_")[1].split('.')[:-1])))
    if len(best_models) == 0:
        best_model_filename = os.path.join(checkpoint_dir,f"best_{accuracy:.4f}.pt")
        torch.save(checkpoint, best_model_filename)
        print(f"Best model saved to {best_model_filename}")
        return
    if len(best_models) < num_best_models or accuracy > parse_checkpoint_acc(best_models[-1]):
        best_model_filename = os.path.join(checkpoint_dir,f"best_{accuracy:.4f}.pt")
        torch.save(checkpoint, best_model_filename)
        print(f"Best model saved to {best_model_filename}")
        # Remove the worst-performing model if there are more than num_best_models
        if len(best_models) >= num_best_models:
            os.remove(best_models[0])

def parse_checkpoint_acc(checkpoint):
    # Split the string using the path separator (either "\\" or "/") to handle different platforms
    parts = checkpoint.split(os.path.sep)
    # Find the "best_*" part in the path
    for part in reversed(parts):
        if part.startswith("best_"):
            best_part = part
            acc = float(best_part.split('_')[1].split('.')[0])
            return acc

def load_checkpoint(model, optimizer, filename):
    """
    Load model checkpoint from a file.
    
    Args:
        model (torch.nn.Module): The PyTorch model to load the weights into.
        optimizer (torch.optim.Optimizer): The optimizer to load the state into.
        filename (str): The path to the checkpoint file.
    
    Returns:
        epoch (int): The last saved epoch.
        loss (float): The loss at the last saved epoch.
        accuracy (float): The accuracy at the last saved epoch.
    """
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    accuracy = checkpoint['accuracy']
    print(f"Checkpoint loaded from {filename}")
    return epoch, loss, accuracy
    