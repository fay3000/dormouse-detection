import argparse
import numpy as np
import os
from typing import Optional
from keras_cv import visualization
from src.model import ObjectDetection
from src.utils import preprocess_image
from src.logger import logging, configure_logging, logger
from config import Config


class Inference:
    def __init__(self, model_path: Optional[str] = None, config: Optional[Config] = None):
        """
        Args:
            model_path: Path to a pre-trained ObjectDetection model. If not provided, 
                        the model can be loaded later by calling `load_model(model_path)`.
            config: Configuration parameters. If not provided, default Config values are used.
        """
        self.config = config if config else Config()
        self.bbox_format = self.config.bbox_format

        # Load the model if a path is provided
        self.model = self.load_model(model_path) if model_path else None

    
    def load_model(self, model_path: str):
        """
        Loads and compiles a trained object detection model.

        Args:
            model_path: Path to the model file. Must have .keras extension.
        """
        try:
            logger.info(f"Loading model: {model_path}")
            detector = ObjectDetection()
            model = detector.load_model(model_path)
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    
    def predict(self, image_path: str, confidence_threshold: float = 0.2):
        """
        Predicts a single image.

        Args:
            image_path: Input image path.
            confidence_threshold: Prediction confidence threshold, default 0.2.
        Returns:
            y_pred (np.ndarray): Model prediction.
            image (np.ndarray): The loaded input image.
        """
        image = self.load_image(image_path)
        # Overwrite model's default confidence threshold
        self.model.prediction_decoder.confidence_threshold = confidence_threshold

        logger.info(f"Predicting Image: {image_path}")
        y_pred = self.model.predict(image)
        return y_pred, image

    
    def load_image(self, image_path: str):
        image, _, _, _, _ = preprocess_image(image_path, self.config.input_shape)
        return np.array([image]) # Shape (1, height, width, channels)

    
    def predict_images_in_folder(self, dir_path: str):
        """
        Performs predictions on all images in the specified folder.
        Results are saved in the subfolder "inference_results" within the given folder.

        Args:
            dir_path: Directory path containing the input images.
        """
        result_dir = os.path.join(dir_path, 'inference_results')
        os.makedirs(result_dir, exist_ok=True)

        for filename in os.listdir(dir_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(dir_path, filename)
                try:
                    self.predict_and_save_image(image_path, result_dir)
                except Exception as e:
                    logger.warning(f"Skipping {image_path} due to an error: {e}")

    
    def predict_and_save_image(self, image_path: str, output_dir: str = None):
        """
        Predicts bounding boxes and class labels for the specified image,
        then saves the resulting image overlaid with the predicted bounding boxes,
        class labels and prediction confidence to the output directory.

        Args:
            image_path: Input image path.
            output_dir: Directory to save the output image.

        Returns:
            y_pred (np.ndarray): Model prediction.
            image (np.ndarray): The image overlaid with predicted bounding boxes.
        """
        y_pred, image = self.predict(image_path)
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        if output_dir is None:
            output_dir = os.path.dirname(image_path)  # Use the same directory as the input image

        result_path = os.path.join(output_dir, f"{image_name}_prediction.png")

        class_ids = self.config.class_ids
        class_mapping = dict(zip(range(len(class_ids)), class_ids))

        # Plot and save the prediction
        visualization.plot_bounding_box_gallery(
            image,
            value_range=(0, 255),
            rows=1,
            cols=1,
            y_pred=y_pred,
            scale=4,
            font_scale=0.7,
            bounding_box_format=self.bbox_format,
            class_mapping=class_mapping,
            path=result_path
        )
        logger.info(f"Saved Prediction: {result_path}")
        return y_pred, image


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run inference on images using a pre-trained model. "
                                                 "Provide the model path and either an image path or image dir.")
    parser.add_argument('--model_path', type=str, required=True, help='The pre-trained model filepath '
                                                                      'in .keras format.'
                                                                   ' Example: --model_path "trained_models/model.keras"')
    parser.add_argument('--image_path', type=str, help='Path to the image to predict.')
    parser.add_argument('--image_dir', type=str, help='Path to a folder containing images to predict.')
    parser.add_argument('--confidence_threshold', type=float, default=0.2,
                        help='Confidence threshold for predictions, default=0.2)')
    return parser.parse_args()


def main(args):
    try:
        inference = Inference(model_path=args.model_path)

        # Predict a single image
        if args.image_path:
            inference.predict_and_save_image(args.image_path)

        # Predict all images in a folder
        if args.image_dir:
            inference.predict_images_in_folder(args.image_dir)

        logger.info("Image prediction completed successfully.")

    except Exception as e:
        logger.error(f"Inference failed: {e}")


if __name__ == '__main__':
    configure_logging(log_level=logging.INFO)
    args = parse_arguments()

    # Check for at least one valid input for prediction
    if args.image_path or args.image_dir:
        main(args)
    else:
        logger.warning(f"Missing argument. Please provide an --image_path or --image_dir.")
