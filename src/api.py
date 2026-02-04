import logging
import numpy as np
import time
from fastapi import FastAPI, HTTPException, Query, status, Response, Depends
from pathlib import Path
from src.model import LogisticRegression
from src.preprocessing import Preprocessor
from src.evaluation import Evaluation
from src.tracking import ExperimentTracker

app = FastAPI(title="Raisin Classification API")
logger = logging.getLogger("uvicorn")
def get_latest_model():
    base_path = Path(__file__).resolve().parent.parent / "models"
    model_files = list(base_path.glob("*.npz"))

    if not model_files:
        raise HTTPException(status_code=404, detail="No model files found in /models")

    latest_model_path = max(model_files, key=lambda p: p.stat().st_mtime)

    try:
        return {"data": np.load(latest_model_path), "path": latest_model_path}
    except Exception as e:
        logger.error("Latest model file not found. Error: {error}".format(error=(e)))
        raise HTTPException(status_code=500, detail=f"Failed to load model")


@app.get("/health")
def health_check(response: Response, model_resource: dict = Depends(get_latest_model)):
    if model_resource is None:
        logger.error("There is no model")
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {
            "model_loaded": False,
            "message": "No model artifact was found"
        }

    return {
        "model_loaded": True,
        "active_model": model_resource["path"].name,
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
        logger.error(f"Training failed: {str(e)}")
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"error": "Training failed", "details": str(e)}

@app.get("/model/info")
def get_model_info(model: dict = Depends(get_latest_model)):
    model_data = model["data"]

    return {
        "active_model": model["path"].name,
        "parameters": {
            "weights": model_data['weights'].tolist(),
            "bias": float(model_data['bias'])
        },
        "normalization": {
            "mean": model_data['mean'].tolist(),
            "std": model_data['std'].tolist()
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