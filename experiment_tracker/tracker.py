# experiment_tracker/tracker.py
from pathlib import Path
import torch
from .database import init_db, Experiment, TrainingMetric, EvaluationMetric


def _convert_paths_to_strings(config):
    """Convert any Path objects in config to strings recursively."""
    if isinstance(config, dict):
        return {k: _convert_paths_to_strings(v) for k, v in config.items()}
    elif isinstance(config, (list, tuple)):
        return [_convert_paths_to_strings(v) for v in config]
    elif isinstance(config, Path):
        return str(config)
    return config


class ExperimentTracker:
    def __init__(
        self, base_artifacts_dir="artifacts", db_url="sqlite:///experiments.db"
    ):
        self.session = init_db(db_url)
        self.base_artifacts_dir = Path(base_artifacts_dir)
        self.base_artifacts_dir.mkdir(exist_ok=True)
        self.current_experiment = None

    def start_experiment(self, name, config):
        """Start tracking a new experiment with given configuration."""

        # Convert any Path objects in config to strings
        serializable_config = _convert_paths_to_strings(config)

        # Create experiment record
        experiment = Experiment(
            name=name,
            config=serializable_config,
        )

        # Commit to database
        self.session.add(experiment)
        self.session.commit()
        self.current_experiment = experiment

        # Create artifacts directory for this experiment
        experiment_dir = self.base_artifacts_dir / str(experiment.id)
        experiment_dir.mkdir(exist_ok=True)
        experiment.artifacts_path = str(experiment_dir)

        # Update the experiment with the artifacts path
        self.session.commit()

        return experiment

    def log_training_metrics(
        self, model, epoch, train_loss, train_accuracy, val_loss, val_accuracy
    ):
        """Log metrics for a training epoch."""
        if not self.current_experiment:
            raise RuntimeError("No active experiment. Call start_experiment first.")

        # Save model checkpoint to local file storage
        filename = f"epoch_{epoch}.pth"
        path = Path(self.current_experiment.artifacts_path) / filename
        torch.save(model.state_dict(), path)

        metric = TrainingMetric(
            experiment_id=self.current_experiment.id,
            epoch=epoch,
            checkpoint_path=str(path),
            train_loss=train_loss,
            train_accuracy=train_accuracy,
            val_loss=val_loss,
            val_accuracy=val_accuracy,
        )
        self.session.add(metric)
        self.session.commit()

    def log_evaluation_metrics(self, dataset_name, loss, accuracy):
        """Log evaluation metrics for a specific dataset."""
        if not self.current_experiment:
            raise RuntimeError("No active experiment. Call start_experiment first.")

        metric = EvaluationMetric(
            experiment_id=self.current_experiment.id,
            dataset_name=dataset_name,
            loss=loss,
            accuracy=accuracy,
        )
        self.session.add(metric)
        self.session.commit()

    def end_experiment(self):
        """End the current experiment."""
        self.current_experiment = None
