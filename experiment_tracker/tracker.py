# experiment_tracker/tracker.py
from pathlib import Path
import json
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
        # Create artifacts directory for this experiment
        experiment_dir = self.base_artifacts_dir / name
        experiment_dir.mkdir(exist_ok=True)

        # Convert any Path objects in config to strings
        serializable_config = _convert_paths_to_strings(config)

        # Create experiment record
        experiment = Experiment(
            name=name, config=serializable_config, artifacts_path=str(experiment_dir)
        )
        self.session.add(experiment)
        self.session.commit()
        self.current_experiment = experiment
        return experiment

    def log_training_metrics(
        self, epoch, train_loss, train_accuracy, val_loss, val_accuracy
    ):
        """Log metrics for a training epoch."""
        if not self.current_experiment:
            raise RuntimeError("No active experiment. Call start_experiment first.")

        metric = TrainingMetric(
            experiment_id=self.current_experiment.id,
            epoch=epoch,
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

    def save_artifact(self, artifact, filename):
        """Save an artifact to the experiment's artifact directory."""
        if not self.current_experiment:
            raise RuntimeError("No active experiment. Call start_experiment first.")

        artifact_path = Path(self.current_experiment.artifacts_path) / filename

        # Handle different types of artifacts
        if isinstance(artifact, dict) and any(
            isinstance(v, torch.Tensor) for v in artifact.values()
        ):
            # This is likely a PyTorch model state dict or checkpoint
            torch.save(artifact, artifact_path)
        elif isinstance(artifact, (dict, list)):
            # Regular JSON-serializable data
            with open(artifact_path, "w") as f:
                json.dump(_convert_paths_to_strings(artifact), f)
        else:
            # Assume it's an object with a save method
            artifact.save(artifact_path)

        return str(artifact_path)

    def end_experiment(self):
        """End the current experiment."""
        self.current_experiment = None
