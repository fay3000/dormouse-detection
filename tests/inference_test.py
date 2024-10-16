""" Test the Inference Class. The working directory must be set to the project root.  """
from inference import Inference
from src.logger import logging, configure_logging, logger

def test():
    try:
        # Initialise the inference instance with a pre-trained model
        model_path = "./trained_models/dormouse_model_31-8_01-32-18_best1.keras"
        inference = Inference(model_path)

        # Predict all images in a specified folder
        image_dir = "./tests/unittest_images"
        inference.predict_images_in_folder(image_dir)

        # Predict a single image
        image_path = "./tests/unittest_images/ginger_cats.png"
        inference.predict_and_save_image(image_path)

        logger.info(f"Inference Test Complete")

    except Exception as e:
        logger.error(f"Inference Test Failed: {e}")


if __name__ == '__main__':
    configure_logging(log_level=logging.INFO)
    test()
