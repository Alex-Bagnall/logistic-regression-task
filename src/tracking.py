import json
from datetime import datetime
from pathlib import Path
import logging
import uuid

logger = logging.getLogger(__name__)

class ExperimentTracker:
    def __init__(self, experiments_dir="../experiments"):
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(exist_ok=True)

    def log_experiment(self, hyperparameters: dict, metrics: dict, model_path: str, seed: int):
        run_id = str(uuid.uuid4())

        experiment = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "hyperparameters": hyperparameters,
            "metrics": metrics,
            "model_path": model_path,
            "seed": seed
        }

        file_path = self.experiments_dir / f"{run_id}.json"

        with open(file_path, "w") as f:
            json.dump(experiment, f, indent=2)

        logger.info(f"Experiment saved to: {file_path}")