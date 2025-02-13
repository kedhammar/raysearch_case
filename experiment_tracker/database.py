# experiment_tracker/database.py
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    Float,
    String,
    JSON,
    ForeignKey,
    DateTime,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import datetime

Base = declarative_base()


class Experiment(Base):
    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    start_time = Column(DateTime, default=datetime.datetime.utcnow)
    config = Column(JSON, nullable=False)
    artifacts_path = Column(String)

    # Relationships
    training_metrics = relationship("TrainingMetric", back_populates="experiment")
    evaluation_metrics = relationship("EvaluationMetric", back_populates="experiment")
    checkpoints = relationship("Checkpoint", back_populates="experiment")


class TrainingMetric(Base):
    __tablename__ = "training_metrics"

    id = Column(Integer, primary_key=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id"))
    epoch = Column(Integer, nullable=False)
    train_loss = Column(Float, nullable=False)
    train_accuracy = Column(Float, nullable=False)
    val_loss = Column(Float, nullable=False)
    val_accuracy = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationship
    experiment = relationship("Experiment", back_populates="training_metrics")
    checkpoint = relationship(
        "Checkpoint", back_populates="training_metric", uselist=False
    )


class Checkpoint(Base):
    __tablename__ = "checkpoints"

    id = Column(Integer, primary_key=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id"))
    path = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    epoch = Column(Integer, ForeignKey("training_metrics.epoch"), nullable=False)

    # Relationship
    experiment = relationship("Experiment", back_populates="checkpoints")
    training_metric = relationship("TrainingMetric", back_populates="checkpoint")


class EvaluationMetric(Base):
    __tablename__ = "evaluation_metrics"

    id = Column(Integer, primary_key=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id"))
    dataset_name = Column(String, nullable=False)
    loss = Column(Float, nullable=False)
    accuracy = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationship
    experiment = relationship("Experiment", back_populates="evaluation_metrics")


def init_db(db_url="sqlite:///experiments.db"):
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()
