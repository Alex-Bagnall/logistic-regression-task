import logging
import numpy as np
import os
from fastapi import FastAPI, HTTPException, Query, status, Response
from pathlib import Path
from src.model import LogisticRegression
from src.preprocessing import Preprocessor
from src.evaluation import Evaluation
from src.tracking import ExperimentTracker

app = FastAPI(title="Raisin Classification API")
logger = logging.getLogger("uvicorn")
MODEL_PATH = None

def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return np.load(MODEL_PATH)

@app.get("/health")
def health_check(response: Response):
    try:
        model_data = load_model()
        response.status = status.HTTP_200_OK
        return { "message": "OK" }
    except Exception as e:
        logger.error("Model is missing: {exception}".format(exception=str(e)))
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"error": "Model artifact missing, please POST to /train endpoint"}

@app.post("/train")
def train_model(response: Response, seed: int = Query(42, description="The random seed for reproducibility")):
    global MODEL_PATH
    logging.basicConfig(level=logging.INFO)
    try:
        preprocessor = Preprocessor()
        tracker = ExperimentTracker()

        features_train, features_test, labels_train, labels_test = preprocessor.preprocess(seed=seed)

        model = LogisticRegression()
        model.fit(features_train, labels_train)

        y_prediction = model.predict(features_test)
        evaluator = Evaluation(y_prediction, labels_test)

        metrics = {
            "accuracy": evaluator.accuracy(),
            "precision": evaluator.precision(),
            "recall": evaluator.recall(),
            "f1": evaluator.f1()
        }

        MODEL_PATH = str(Path(__file__).resolve().parent.parent / "models" /  "model_{seed}.npz".format(seed=seed))

        np.savez(MODEL_PATH,
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
            model_path=MODEL_PATH,
            seed=seed
        )

        return {
            "message": "Training complete",
            "experiment_id": run_id,
            "seed_used": seed,
            "metrics": metrics,
            "artifact_path": MODEL_PATH
        }

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"error": "Training failed", "details": str(e)}