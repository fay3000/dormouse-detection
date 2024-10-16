""" Test the DataLoader class. The working directory must be set to the project root. """
from keras_cv import visualization
from config import Config
from src.data_loader import DataLoader
from src.logger import logging, configure_logging, logger


def test():
    try:
        # Define config parameters
        config = Config()

        # Initialise the data loader
        data_loader = DataLoader(config)

        # Load and prepare the dataset
        train_images, test_images, train_labels, test_labels = data_loader.prepare_data(augment=True)

        # Visualise the dataset
        class_ids = config.class_ids
        class_mapping = dict(zip(range(len(class_ids)), class_ids))

        def plot(images, labels):
            # Plots a 3x3 grid of images with their bboxes
            visualization.plot_bounding_box_gallery(
                images,
                value_range=(0, 255),
                y_true=labels,
                rows=6,
                cols=6,
                scale=2.4,
                font_scale=0.7,
                bounding_box_format=data_loader.bbox_format,
                class_mapping=class_mapping,
                show=True
            )

        plot(train_images, train_labels)
        plot(test_images, test_labels)

    except Exception as e:
        logger.error(f"Inference Test Failed: {e}")


if __name__ == '__main__':
    configure_logging(log_level=logging.INFO)
    test()