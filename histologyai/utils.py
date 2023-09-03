import os
import time
import torch
import glob
import importlib
    
def convert_path(path):
    if os.name == 'nt':  # Windows
        return path.replace('/', '\\')
    else:  # Linux
        return path.replace('\\', '/')


class ExperimentManager:
    def __init__(self, exp_name):
        self.experiment_name = self._generate_experiment_name(exp_name)
        self.experiment_dir = os.path.join("logs", "checkpoints", self.experiment_name)
        self.best_model_path = ""
        self.best_acc = -99999
        os.makedirs(self.experiment_dir, exist_ok=True)

    def _generate_experiment_name(self, exp_name):
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        return f"{exp_name}_{timestamp}"
    
    def save_checkpoint(self, checkpoint_dict:dict, num_best_models=5):
        """Save checkpoint to a file based on a certain frequency with generated checkpoint name.

        Parameters
        ----------
        checkpoint_dict:
            model : torch.nn
                Torch model
            optimizer : torch.optimizer
                Torch optimizer
            epoch : int
                Current epoch
            loss : float
                Current training loss
            accuracy : float
                Validation Acc
            architecture : string
                Model architecture name(class name)
            classes : list
                List of classes in dataset
        """
        epoch = checkpoint_dict["epoch"]
        checkpoint_filename = f"epoch_{epoch}.pt"
        checkpoint_path = os.path.join(self.experiment_dir, checkpoint_filename)    
        checkpoint_dir = os.path.dirname(checkpoint_path)
        accuracy = checkpoint_dict["val_accuracy"]
        files_in_dir = glob.glob(os.path.join(checkpoint_dir, 'best_*.pt'))
        best_models = sorted(files_in_dir, key=lambda x: float('.'.join(\
            os.path.basename(x).split("_")[1].split('.')[:-1])))
        if len(best_models) == 0:
            best_model_filename = os.path.join(checkpoint_dir,f"best_{accuracy:.4f}.pt")
            torch.save(checkpoint_dict, best_model_filename)
            self.best_model_path = best_model_filename
            print(f"Best model saved to {best_model_filename}")
            return best_model_filename
        if len(best_models) < num_best_models or accuracy > parse_checkpoint_acc(best_models[-1]):
            best_model_filename = os.path.join(checkpoint_dir,f"best_{accuracy:.4f}.pt")
            torch.save(checkpoint_dict, best_model_filename)
            print(f"Best model saved to {best_model_filename}")
            self.best_model_path = best_model_filename
            # Remove the worst-performing model if there are more than num_best_models
            if len(best_models) >= num_best_models:
                os.remove(best_models[0])
            return best_model_filename

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

def create_module(model_arch:str, num_classes:int):
    """Dynamically import the custom model class"""
    custom_model_module = importlib.import_module(f"histologyai.models.{model_arch}")
    ModelClass = getattr(custom_model_module, model_arch)
    model = ModelClass(num_classes=num_classes)
    return model