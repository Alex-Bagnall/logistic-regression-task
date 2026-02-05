import logging
import numpy as np
import time
import json
from fastapi import FastAPI, HTTPException, Query, status, Response
from typing import Optional
from pathlib import Path
from src.model import LogisticRegression
from src.preprocessing import Preprocessor
from src.evaluation import Evaluation
from src.tracking import ExperimentTracker
from pydantic import BaseModel

app = FastAPI(title="Raisin Classification API")
logger = logging.getLogger("uvicorn")

class PredictionRequest(BaseModel):
    area: float
    major_axis: float
    minor_axis_length: float
    eccentricity: float
    convex_area: float
    extent: float
    perimeter: float


def get_model_resource(seed: Optional[int] = None):
    base_path = Path(__file__).resolve().parent.parent / "models"
    model_files = list(base_path.glob("*.npz"))

    if not model_files:
        raise HTTPException(status_code=404, detail="No model files found in /models")

    if seed is not None:
        target_file = base_path / "model_{seed}.npz".format(seed=seed)
        if not target_file.exists():
            raise HTTPException(
                status_code=404,
                detail="No model found for seed {seed}".format(seed=seed)
            )
        model_path = target_file
    else:
        if not model_files:
            return None
        model_path = max(model_files, key=lambda p: p.stat().st_mtime)

    try:
        return {"data": np.load(model_path), "path": model_path}
    except Exception as e:
        logger.error("Latest model file not found. Error: {error}".format(error=(e)))
        raise HTTPException(status_code=500, detail="Failed to load model")

def label_features(weights: list[float]) -> dict:
    feature_names = ["area", "major_axis", "minor_axis_length", "eccentricity", "convex_area", "extent", "perimeter"]
    return {name: round(weight, 4) for name, weight in zip(feature_names, weights)}

@app.get("/health")
def health_check():
    return {
        "message": "OK"
    }

@app.post("/train")
def train_model(response: Response, seed: int = Query(42, description="The random seed for reproducibility")):
    try:
        preprocessor = Preprocessor()
        tracker = ExperimentTracker()

        features_train, features_test, labels_train, labels_test = preprocessor.preprocess(seed=seed)

        model = LogisticRegression()
        model.fit(features_train, labels_train)

        y_prediction = model.predict(features_test)
        evaluator = Evaluation(y_prediction, labels_test)

        model_dir = Path(__file__).resolve().parent.parent / "models"
        model_dir.mkdir(exist_ok=True)

        model_path = model_dir / "model_{seed}.npz".format(seed=seed)

        metrics = {
            "accuracy": evaluator.accuracy(),
            "precision": evaluator.precision(),
            "recall": evaluator.recall(),
            "f1": evaluator.f1()
        }

        np.savez(model_path,
                 weights=model.weights,
                 bias=model.bias,
                 mean=preprocessor.mean,
                 std=preprocessor.std)

        run_id = tracker.log_experiment(
            hyperparameters={
                "learning_rate": model.learning_rate,
                "epochs": model.epochs,
                "lambda_reg": model.lambda_reg,
                "seed": seed
            },
            metrics=metrics,
            model_path=str(model_path),
            seed=seed
        )

        return {
            "message": "Training complete",
            "experiment_id": run_id,
            "seed_used": seed,
            "metrics": metrics,
            "artifact_path": model_path
        }

    except Exception as e:
        logger.error("Training failed: {error}".format(error=str(e)))
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"error": "Training failed", "details": str(e)}

@app.get("/model/info")
def get_model_info(seed: Optional[int] = None):
    model = get_model_resource(seed=seed)
    model_data = model["data"]

    return {
        "active_model": model["path"].name,
        "parameters": {
            "weights": label_features(model_data['weights'].tolist()),
            "bias": float(model_data['bias'])
        },
        "normalization": {
            "mean": label_features(model_data['mean'].tolist()),
            "std": label_features(model_data['std'].tolist())
        }
    }

@app.get("/models")
def list_all_models():
    model_dir = Path(__file__).resolve().parent.parent / "models"

    if not model_dir.exists():
        return {"models": [], "count": 0}

    model_list = []
    for file in model_dir.glob("*.npz"):
        stats = file.stat()
        model_list.append({
            "filename": file.name,
            "created_at": time.ctime(stats.st_ctime),
            "size_kb": round(stats.st_size / 1024, 2),
            "is_latest": False
        })

    model_list.sort(key=lambda x: x["created_at"], reverse=True)

    if model_list:
        model_list[0]["is_latest"] = True

    return {
        "models": model_list,
        "total_count": len(model_list),
        "directory": str(model_dir.absolute())
    }

@app.post("/predict")
def predict(request: PredictionRequest, seed: Optional[int] = None):
    features = [
        request.area,
        request.major_axis,
        request.minor_axis_length,
        request.eccentricity,
        request.convex_area,
        request.extent,
        request.perimeter
    ]

    model = get_model_resource(seed=seed)
    model_data = model["data"]

    weights = model_data['weights']
    bias = float(model_data['bias'])
    mean = model_data['mean']
    std = model_data['std']

    features_norm = (np.array(features) - mean) / std

    z = np.dot(features_norm, weights) + bias
    probability = float(1 / (1 + np.exp(-z)))
    prediction = 1 if probability >= 0.5 else 0

    logger.info("Prediction: {prediction}, Probability: {probability:.4f}".format(prediction=prediction, probability=probability))

    return {
        "prediction": prediction,
        "probability": round(probability, 4),
        "label": "Besni" if prediction == 1 else "Kecimen",
        "model_used": model["path"].name
    }

@app.get("/explain")
def explain(seed: Optional[int] = None):
    feature_names = ["area", "major_axis", "minor_axis_length", "eccentricity", "convex_area", "extent", "perimeter"]

    model = get_model_resource(seed)
    model_data = model["data"]
    weights = model_data['weights']

    feature_importance = sorted(
        zip(feature_names, weights.tolist()),
        key=lambda item: abs(item[1]),
        reverse=True
    )

    return {
        "model_used": model["path"].name,
        "feature_importance": [
            {
                "rank": featureRank + 1,
                "feature": name,
                "weight": round(weight, 4),
                "direction": "Besni" if weight > 0 else "Kecimen",
                "impact": "high" if abs(weight) > 1 else "medium" if abs(weight) > 0.5 else "low"
            }
            for featureRank, (name, weight) in enumerate(feature_importance)
        ]
    }

@app.get("/experiments/best")
def get_best_experiment(metric: str = Query("accuracy", enum=["accuracy", "f1", "precision", "recall"])):
    experiments_dir = Path(__file__).resolve().parent.parent / "experiments"
    files = list(experiments_dir.glob("*.json"))
    best_run = None
    highest_score = -1.0

    for file_path in files:
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

                metrics = data.get("metrics", {})
                current_score = metrics.get(metric, -1.0)

                if current_score > highest_score:
                    highest_score = current_score
                    best_run = data
        except (json.JSONDecodeError, IOError) as e:
            logger.warning("Skipping invalid experiment file {file_path.name}: {e}".format(file_path=file_path, e=e))
            continue

    if not best_run:
        raise HTTPException(
            status_code=404,
            detail="No experiments found with metric: {metric}".format(metric=metric)
        )

    best_seed = best_run.get("seed")
    docker_model_path = "/app/models/model_{best_seed}.npz".format(best_seed=best_seed)

    return {
        "best_metric": metric,
        "score": highest_score,
        "seed": best_seed,
        "docker_model_path": docker_model_path,
        "metadata": {
            "experiment_id": best_run.get("experiment_id"),
            "timestamp": best_run.get("timestamp")
        },
        "full_details": best_run
    }