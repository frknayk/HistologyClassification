import logging
import os
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import wandb
import coloredlogs


from munch import DefaultMunch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    f1_score, 
    roc_curve, 
    auc, 
    precision_recall_curve)

from histologyai.utils import ExperimentManager, create_module
from histologyai.dataset_loader import ImageClassificationDataset, InferenceDataset
from histologyai.evaluate import log_inference_result, log_test_result

# Configure the logger
coloredlogs.install(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClassificationTrainer:
    def __init__(self, config:dict):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.config = DefaultMunch.fromDict(config)
        # If early-stop is given by config, check validation loss history as criteria
        if self.config.training.early_stop:
            self.val_lost_list = []
        # Create image classification dataset
        self.image_classification_dataset = ImageClassificationDataset(self.config)
        self.train_loader, self.val_loader, self.test_loader, self.classes = \
            self.image_classification_dataset.create_data_loaders(
            batch_size_train=self.config.training.batch_size,
            batch_size_eval=self.config.evaluation.batch_size)

    def _initialize_optimizer(self, model):
        optimizer_config = self.config.optimizer_config
        if optimizer_config.type == "SGD":
            optimizer = optim.SGD(
                model.parameters(), 
                lr=optimizer_config.learning_rate, 
                momentum=optimizer_config.momentum)
        elif optimizer_config.type == "Adam":
            optimizer = optim.Adam(model.parameters(), 
                lr=optimizer_config.learning_rate, 
                betas=optimizer_config.betas)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_config.optimizer_type}")
        return optimizer

    def _initialize_model(self):
        model = create_module(
            model_arch=self.config.model.architecture,
            num_classes=self.config.data.num_classes)
        model.to(self.device)
        return model

    def eval_model(self, model, criterion, dataset_loader):
        """Evaluate model, calculate metrics

        Parameters
        ----------
        model : PyTorch model (torch.nn)
            _description_
        criterion : Loss function
            _description_
        dataset_loader : Pytorch Dataset Loader
            _description_

        Returns
        -------
        dict
            dict of metrics
        """
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        running_loss = 0.0
        running_corrects = 0
        with torch.no_grad():
            for inputs, labels in dataset_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        all_preds = np.asarray(all_preds)
        all_labels = np.asarray(all_labels)
        all_probs = np.asarray(all_probs)
        eval_metrics = {
            "running_loss": running_loss,
            "running_corrects": running_corrects,
            "all_preds": all_preds,
            "all_labels": all_labels,
            "all_probs": all_probs
        }
        return DefaultMunch.fromDict(eval_metrics)

    def calculate_log_metrics_val(self, model, criterion):
        metrics = self.eval_model(model, criterion, self.val_loader)
        val_acc = metrics.running_corrects.double().item() / len(self.val_loader.sampler)
        val_loss = metrics.running_loss / len(self.val_loader.sampler)
        f1 = f1_score(metrics.all_labels, metrics.all_preds, average='weighted')
        # Calculate FPR,TPR,PR,Recall for each class
        validation_metrics = {
            "val_loss":val_loss,
            "val_acc":val_acc,
            "val_f1_score":f1
        }
        wandb.log(validation_metrics)
        return val_loss, val_acc, f1

    def calculate_log_metrics_test(self, model, criterion):
        if self.test_loader is None:
            return
        metrics = self.eval_model(model, criterion, self.test_loader)
        logger.info("Calculating metrics for test dataset..")
        test_loss = metrics.running_loss / len(self.test_loader.sampler)
        test_acc = metrics.running_corrects.double().item() / len(self.test_loader.sampler)
        f1 = f1_score(metrics.all_labels, metrics.all_preds, average='weighted')
        wandb.log({"test_loss":test_loss,
                    "test_acc":test_acc,
                    "test_f1_score":f1})
        # Metrics belonging to each class
        classes_sorted = [str(np_str) for np_str in list(np.sort(self.classes))]
        class_modified = [classes_sorted + ["PR_or_ROC"]][0]
        class_metrics_roc_auc = []
        class_metrics_pr_auc = []
        for cl_idx, _ in enumerate(classes_sorted):
            fpr, tpr, _ = roc_curve(metrics.all_labels, metrics.all_probs[:, cl_idx], pos_label=cl_idx)
            roc_auc = auc(fpr, tpr)
            precision, recall, _ = precision_recall_curve(metrics.all_labels, metrics.all_probs[:, cl_idx], pos_label=cl_idx)
            pr_auc = auc(recall, precision)
            class_metrics_roc_auc.append(roc_auc)
            class_metrics_pr_auc.append(pr_auc)
        class_metrics_roc_auc.append("ROC")
        class_metrics_pr_auc.append("PR")
        table_auc_pr = wandb.Table(columns=class_modified, data=[class_metrics_roc_auc,class_metrics_pr_auc])
        wandb.log({"AUC(ROC and PR) by Each Class": table_auc_pr})
        wandb.log({'TEST ROC-AUC': wandb.plot.roc_curve(metrics.all_labels, metrics.all_probs, self.classes)})
        wandb.log({'TEST PR-AUC': wandb.plot.pr_curve(metrics.all_labels, metrics.all_probs, self.classes)})
        cm = wandb.plot.confusion_matrix(y_true=metrics.all_labels, preds=metrics.all_preds, class_names=self.classes)
        wandb.log({"Confusion Matrix":cm})

    def train(self):
        def core(model, criterion, optimizer, experiment_manager):   
            for epoch in range(self.config.training.epochs):
                # Training phase
                model.train()
                running_loss = 0.0
                running_corrects = 0
                for inputs, labels in self.train_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    _, preds = torch.max(outputs, 1)
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                train_loss = running_loss / len(self.train_loader.sampler)
                train_acc = running_corrects.double().item() / len(self.train_loader.sampler)
                wandb.log({
                    "train_loss":train_loss,
                    "train_acc":train_acc})
                val_loss, val_acc, f1 = self.calculate_log_metrics_val(model, criterion)
                experiment_manager.save_checkpoint({
                    "epoch": epoch+1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "training_loss": train_loss,
                    "val_accuracy": val_acc,
                    "architecture": self.config.model.architecture,
                    "num_classes": len(self.classes),
                    "classes": self.classes})
                # Use the logger to display colored messages
                logger.info(
                    f"Epoch [{epoch+1}/{self.config.training.epochs}] - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                    f"F1 Score: {f1:.4f}"
                )
                if self.config.early_stop:
                    self.val_lost_list.append(val_loss)
                    if early_stop(self.val_lost_list, patience=10):
                        logger.critical("Early-stop detected, aborting train...")
                        return
            self.calculate_log_metrics_test(model, criterion)
            log_inference_result(model, 
                self.image_classification_dataset.inference_loader, 
                self.device, self.classes)
            # # Load the best model according to validation accuracy
            # logger.warning("Loading best model according to validation accuracy metric..")
            # checkpoint_dict = torch.load(experiment_manager.best_model_path)
            # best_model = create_module(
            #     model_arch=self.config.model.architecture,
            #     num_classes=self.config.data.num_classes)
            # best_model.to(self.device)
            # best_model.load_state_dict(checkpoint_dict["model_state_dict"])
            # log_test_result(best_model, self.test_loader, self.device, self.classes)

        model = self._initialize_model()
        criterion = nn.CrossEntropyLoss()
        optimizer = self._initialize_optimizer(model)
        experiment_manager = ExperimentManager(self.config.experiment_name)
        # Initialize Weights and Biases
        wandb.init(project="Histology" ,name=experiment_manager.experiment_name, 
            config=self.config, dir="logs_deneme")
        core(model,criterion,optimizer, experiment_manager)
        logger.warning("Training is finished with following results:")


def early_stop(val_loss_list, patience=5, delta=0.0):
    """
    Implement early stopping based on validation loss.
    
    Args:
        val_loss_list (list): The validation loss at the current epoch.
        patience (int): Number of epochs with no improvement after which training will be stopped.
        delta (float): Minimum change in the validation loss to be considered as improvement.
        
    Returns:
        stop (bool): True if training should be stopped, False otherwise.
    """
    if len(val_loss_list) < patience + 1:
        return False
    recent_losses = val_loss_list[-patience:]
    best_loss = min(val_loss_list)
    if np.mean(recent_losses) - best_loss > delta:
        return True
    return False
    

if __name__ == "__main__":
    # Load configuration from YAML and initiate training
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", 
        default="resnet50.yaml", 
        help="Name of the configuration YAML file")
    args = parser.parse_args()
    # Get the path to the config folder under the histologyai package
    config_folder = os.path.join(os.path.dirname(__file__), 'configs')
    # Load the configuration file
    config_file_path = os.path.join(config_folder, args.config)
    with open(config_file_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    # Run classification
    trainer = ClassificationTrainer(config)
    trainer.train()
