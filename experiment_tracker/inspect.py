# experiment_tracker/inspect.py
import pandas as pd
from experiment_tracker.tracker import Experiment, TrainingMetric, EvaluationMetric


class DBInspector:
    def __init__(self, session):
        self.session = session

    def list_experiments(self):
        """Show a summary of all experiments."""
        experiments = self.session.query(Experiment).all()

        rows = []
        for exp in experiments:
            # Get latest metrics TODO
            latest_train = (
                self.session.query(TrainingMetric)
                .filter_by(experiment_id=exp.id)
                .order_by(TrainingMetric.epoch.desc())
                .first()
            )

            eval_metrics = (
                self.session.query(EvaluationMetric)
                .filter_by(experiment_id=exp.id)
                .all()
            )

            row = {
                "ID": exp.id,
                "Name": exp.name,
                "Start Time": exp.start_time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            row.update(exp.config)

            rows.append(row)

        df = pd.DataFrame(rows)
        return df

    def get_properties(self, experiment_id):
        """Get a specific experiment by ID."""
        exp = self.session.query(Experiment).filter_by(id=experiment_id).first()
        row = {
            "ID": exp.id,
            "Name": exp.name,
            "Start Time": exp.start_time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        return pd.DataFrame([row])

    def get_parameters(self, experiment_id):
        """Get the parameters of a specific experiment."""
        exp = self.session.query(Experiment).filter_by(id=experiment_id).first()
        return pd.DataFrame([exp.config])

    def get_training_metrics(self, experiment_id):
        """Get training metrics history for a specific experiment."""
        metrics = (
            self.session.query(TrainingMetric)
            .filter_by(experiment_id=experiment_id)
            .order_by(TrainingMetric.epoch)
            .all()
        )

        rows = [
            {
                "Epoch": m.epoch,
                "Train Loss": f"{m.train_loss:.4f}",
                "Train Acc": f"{m.train_accuracy:.4f}",
                "Val Loss": f"{m.val_loss:.4f}",
                "Val Acc": f"{m.val_accuracy:.4f}",
                "Timestamp": m.timestamp.strftime("%H:%M:%S"),
            }
            for m in metrics
        ]

        return pd.DataFrame(rows)

    def get_evaluation_metrics(self, experiment_id):
        """Get all evaluation metrics for a specific experiment."""
        metrics = (
            self.session.query(EvaluationMetric)
            .filter_by(
                experiment_id=experiment_id,
            )
            .all()
        )

        rows = {
            m.dataset_name: {
                "Loss": f"{m.loss:.4f}",
                "Acc": f"{m.accuracy:.4f}",
            }
            for m in metrics
        }

        df = pd.DataFrame.from_dict(rows, orient="index")

        return df

    def get_experiment_details(self, experiment_id):
        """Get detailed information about a specific experiment."""
        exp = self.session.query(Experiment).filter_by(id=experiment_id).first()
        if not exp:
            return None

        print(f"Experiment {exp.id}: {exp.name}")
        print("\nConfiguration:")
        for key, value in exp.config.items():
            print(f"  {key}: {value}")

        eval_metrics = (
            self.session.query(EvaluationMetric).filter_by(experiment_id=exp.id).all()
        )

        if eval_metrics:
            print("\nEvaluation Results:")
            for metric in eval_metrics:
                print(f"  {metric.dataset_name}:")
                print(f"    Loss: {metric.loss:.4f}")
                print(f"    Accuracy: {metric.accuracy:.4f}")
