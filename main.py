from src.logger import logging, configure_logging, logger
from src.training_pipeline import TrainingPipeline
from config import Config


def main(config: Config):
    """
    Runs the training pipeline: load data, train model, evaluate model.

    Args:
        config (Config): Configuration parameters for training.
    """
    try:
        pipeline = TrainingPipeline(config)
        train_images, test_images, train_labels, test_labels = pipeline.load_data()
        pipeline.train(train_images, test_images, train_labels, test_labels)
        pipeline.evaluate(test_images, test_labels)

    except Exception as e:
        logger.error(f"Error running pipeline: {e}")
        raise


if __name__ == '__main__':
    configure_logging(log_level=logging.INFO)
    config = Config()
    main(config)
