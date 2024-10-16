from src.logger import logger

class Config:
    """
      Configuration parameters for training the model.
    """
    
    def __init__(self):
        # Parameters for training. Users can modify their values before running main.
        self.input_shape = (224, 224, 3)    # Image dimensions: 224x224, channels=3
        self.batch_size = 16                # Number of samples used for each model update
        self.epochs = 1                    # Number of training epochs
        self.augmentation = True            # Enables training-data augmentation
        self.early_stopping = True          # Enables training to stop early
        self.early_stop_patience = 15       # Stops if model doesn't improve for this many epochs

        # Parameters that are tied to the dataset (only change if the dataset changes)
        self.num_classes = 2
        self.class_ids = ["hazel_dormouse", "none"]  # Indices correspond to integer class_id in the dataset annotations
        self.dataset_dir = "./training_data/hazel_dormouse/obj_train_data"   # Path to the training and test data
        self.saved_model_path = "./trained_models/dormouse_model_31-8_01-32-18_best1.keras"

        # Parameters that are fixed and should not be modified without code changes
        self.bbox_format = "center_xywh"


    def log_training_params(self):
        params = {"input_shape": self.input_shape, "num_classes": self.num_classes,
                  "batch_size": self.batch_size,  "epochs": self.epochs, "augmentation": self.augmentation,
                  "early_stopping": self.early_stopping, "early_stop_patience": self.early_stop_patience}
        logger.info(f"Training Parameters: {params}")
