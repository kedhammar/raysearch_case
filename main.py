from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from experiment_tracker.database import init_db, Experiment
from experiment_tracker.inspect import DBInspector
import matplotlib
import matplotlib.pyplot as plt
import io
import base64

# Hardware constraints for matplotlib
matplotlib.use("Agg")

# Initialize FastAPI application
app = FastAPI(
    title="Experiment Tracker API", description="API for managing experiments"
)

# Initialize database connection
db_session = init_db()
inspector = DBInspector(db_session)


@app.on_event("shutdown")
async def shutdown_event():
    """Close database connection on application shutdown"""
    db_session.close()


# API responses
@app.get("/api/experiments/")
def read_experiments():
    """Retrieve all experiments"""
    return db_session.query(Experiment).all()


@app.get("/api/experiments/{experiment_id}")
def read_experiment(experiment_id: int):
    """Retrieve a specific experiment"""
    experiment = db_session.query(Experiment).get(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return experiment


@app.get("/api/experiments/{experiment_id}/training-metrics")
def read_training_metrics(experiment_id: int):
    """Retrieve training metrics for an experiment"""
    experiment = db_session.query(Experiment).get(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return experiment.training_metrics


@app.get("/api/experiments/{experiment_id}/evaluation-metrics")
def read_evaluation_metrics(experiment_id: int):
    """Retrieve evaluation metrics for an experiment"""
    experiment = db_session.query(Experiment).get(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return experiment.evaluation_metrics


# HTML responses
@app.get("/experiments", response_class=HTMLResponse)
def tabulate_experiments():
    df = inspector.list_experiments()
    # Turn ID cells into hyperlinks to improve navigation
    df["ID"] = df["ID"].apply(lambda x: f'<a href="/experiments/{x}">{x}</a>')
    return df.to_html(index=False, render_links=True, escape=False)


@app.get("/experiments/{experiment_id}", response_class=HTMLResponse)
def tabulate_experiment(experiment_id: int):
    html_contents = "\n".join(
        [
            tabulate_properties(experiment_id),
            tabulate_parameters(experiment_id),
            tabulate_evaluation_metrics(experiment_id),
            tabulate_training_metrics(experiment_id),
            make_plots(experiment_id),
        ]
    )
    return html_contents


@app.get("/experiments/{experiment_id}/properties", response_class=HTMLResponse)
def tabulate_properties(experiment_id: int):
    df = inspector.get_properties(experiment_id)
    html_contents = f"<h1>Experiment {experiment_id}</h1>\n{df.to_html(index=False)}"
    return html_contents


@app.get("/experiments/{experiment_id}/parameters", response_class=HTMLResponse)
def tabulate_parameters(experiment_id: int):
    df = inspector.get_parameters(experiment_id)
    html_contents = f"<h1>Parameters</h1>\n{df.to_html(index=False)}"
    return html_contents


@app.get("/experiments/{experiment_id}/training-metrics", response_class=HTMLResponse)
def tabulate_training_metrics(experiment_id: int):
    df = inspector.get_training_metrics(experiment_id)
    html_contents = f"<h1>Training Metrics</h1>\n{df.to_html(index=False)}"
    return html_contents


@app.get("/experiments/{experiment_id}/evaluation-metrics", response_class=HTMLResponse)
def tabulate_evaluation_metrics(experiment_id: int):
    df = inspector.get_evaluation_metrics(experiment_id)
    html_contents = f"<h1>Evaluation Metrics</h1>\n{df.to_html()}"
    return html_contents


@app.get("/experiments/{experiment_id}/plots", response_class=HTMLResponse)
def make_plots(experiment_id: int):
    df = inspector.get_training_metrics(experiment_id)
    df = df.astype(
        {
            "Epoch": int,
            "Train Loss": float,
            "Train Acc": float,
            "Val Loss": float,
            "Val Acc": float,
        }
    )

    # Create the first figure for Train Loss and Val Loss
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(
        df["Epoch"], df["Train Loss"], label="Train Loss", color="blue", marker="o"
    )
    ax1.plot(df["Epoch"], df["Val Loss"], label="Val Loss", color="red", marker="o")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Train Loss and Val Loss vs Epoch")
    ax1.legend()
    ax1.set_ylim(0, max(df["Val Loss"].max(), df["Train Loss"].max()) * 1.1)

    # Create the second figure for Train Acc and Val Acc
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.plot(df["Epoch"], df["Train Acc"], label="Train Acc", color="green", marker="o")
    ax2.plot(df["Epoch"], df["Val Acc"], label="Val Acc", color="purple", marker="o")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Train Acc and Val Acc vs Epoch")
    ax2.legend()
    ax2.set_ylim(0, 1)

    # Save the first plot to a string buffer
    buf1 = io.BytesIO()
    fig1.savefig(buf1, format="png")
    buf1.seek(0)
    img_str1 = base64.b64encode(buf1.read()).decode("utf-8")
    plt.close(fig1)

    # Save the second plot to a string buffer
    buf2 = io.BytesIO()
    fig2.savefig(buf2, format="png")
    buf2.seek(0)
    img_str2 = base64.b64encode(buf2.read()).decode("utf-8")
    plt.close(fig2)

    # Create HTML content with embedded images
    html_content = f"""
    <html>
    <body>
        <h1>Training Metrics Plots for Experiment {experiment_id}</h1>
        <h2>Train Loss and Val Loss vs Epoch</h2>
        <img src="data:image/png;base64,{img_str1}" />
        <h2>Train Acc and Val Acc vs Epoch</h2>
        <img src="data:image/png;base64,{img_str2}" />
    </body>
    </html>
    """
    return html_content
