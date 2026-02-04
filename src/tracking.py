import json
from datetime import datetime
from pathlib import Path
import logging
import uuid

logger = logging.getLogger(__name__)

class ExperimentTracker:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent.parent
        self.experiments_dir = self.base_dir / "experiments"
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

    def log_experiment(self, hyperparameters: dict, metrics: dict, model_path: str, seed: int):
        experiment_id = str(uuid.uuid4())

        experiment = {
            "experiment_id": experiment_id,
            "timestamp": datetime.now().isoformat(),
            "hyperparameters": hyperparameters,
            "metrics": metrics,
            "model_path": model_path,
            "seed": seed
        }

        file_path = self.experiments_dir / "{experiment_id}.json".format(experiment_id=experiment_id)

        with open(file_path, "w") as f:
            json.dump(experiment, f, indent=2)

        logger.info("Experiment saved to: {file_path}".format(file_path=file_path))
        return experiment_id