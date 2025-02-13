from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import json
from experiment_tracker.database import (
    init_db,
    Experiment,
    TrainingMetric,
    EvaluationMetric,
)

# Initialize FastAPI application
app = FastAPI(
    title="Experiment Tracker API", description="API for managing experiments"
)

# Initialize database connection
db_session = init_db()


@app.on_event("shutdown")
async def shutdown_event():
    """Close database connection on application shutdown"""
    db_session.close()


# Routes
@app.get("/experiments/")
def read_experiments():
    """Retrieve all experiments"""
    return db_session.query(Experiment).all()


@app.get("/experiments/{experiment_id}")
def read_experiment(experiment_id: int):
    """Retrieve a specific experiment"""
    experiment = db_session.query(Experiment).get(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return experiment


@app.get("/experiments/{experiment_id}/training-metrics")
def read_training_metrics(experiment_id: int):
    """Retrieve training metrics for an experiment"""
    experiment = db_session.query(Experiment).get(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return experiment.training_metrics


@app.get("/experiments/{experiment_id}/evaluation-metrics")
def read_evaluation_metrics(experiment_id: int):
    """Retrieve evaluation metrics for an experiment"""
    experiment = db_session.query(Experiment).get(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return experiment.evaluation_metrics
