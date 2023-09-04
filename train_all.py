import subprocess

# Define the Conda environment name
conda_env = "seg"

# Uncomment for selecting all experiments
# # Get the path to the config folder under the histologyai package
# config_files = []
# config_folder = os.path.join(os.path.dirname(__file__), 'configs')
# for filename in os.listdir(config_folder):
#     if filename.endswith(".yaml"):
#         if filename == "hyperparameter_tuning.yaml":
#             continue
#         config_files.append(filename)

# List of configuration files to run
config_files = [
    # "resnet50_adam.yaml",
    # "resnet50_sgd.yaml",
    # "resnet18_adam.yaml",
    # "resnet18_sgd.yaml",
    "baseClassifier_adam.yaml",
    "deepClassifier_adam.yaml"
]

# Activate the Conda environment
activate_command = f"conda activate {conda_env}"
subprocess.run(activate_command, shell=True)

# Loop through the configuration files and run the experiments
for config_file in config_files:
    run_command = f"python histologyai/train_classifier.py --config {config_file}"
    subprocess.run(run_command, shell=True)

# Deactivate the Conda environment (optional, depending on your use case)
deactivate_command = "conda deactivate"
subprocess.run(deactivate_command, shell=True)
