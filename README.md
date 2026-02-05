# Logistic regression task

This is an attempt to write a logistic regression model from scratch using only Python3.14 and numpy for the data processing and prediction.
This was written using the [Raisin binary classification dataset](https://www.kaggle.com/datasets/nimapourmoradi/raisin-binary-classification/data)
## How to run the app
Simply run `docker compose up --build` in the project root and the app will start on port 8000

### In case of emergency
If you find that the container starts but you still can't see the models, it might be due to cached volumes. If you changed your folder structure after running Docker for the first time, Docker sometimes holds onto an old, empty version of that volume.

If things get weird, run this to "nuke" the cache and start fresh:
```bash
docker compose down -v && docker compose up --build
```

## API
The project includes a FastAPI wrapper that serves the model for real-time inference and remote training.

The API will be available at: http://127.0.0.1:8000

For documentation access go to: http://127.0.0.1:8000/docs to test the endpoints via Swagger.

Trained models and experiment logs are automatically synced between the container and your local `./models` and `./experiments` folders.

## Testing
Run tests with:
```bash
pytest tests/ -v
```

## Explanation of files
<details>
    <summary>data_loader.py</summary>
    
The `/data` folder contains the `raisin_dataset.csv` downloaded from the kaggle link above. 
The DataLoader class takes in the filepath on initialisation, this allows the load method to avoid requiring any params.
The load method creates the label_map which is used when reading the csv to convert the `Class` column into '1' or '0' depending on whether the value in each row is 'Besni' or 'Kecimen" i.e. `label_map = {"Kecimen": 0, "Besni": 1}`
The load method skips over the headers, separates the data into features (the first 7 columns of the data) & labels (the final column of the data converted into binary) to prepare the data for preprocessing
</details>

<details>
    <summary>preprocessing.py</summary>

This file contains the `Preprocessor` whose main job is to load the data into the `DataLoader` cass and return a set of feature & label lists for training & testing.
It achieves this by taking the features & labels returned from the `load()` method, normalising the features using Z-score noramlisation, and then using permutation to create a new numpy array which is shuffled. 
This shuffled array of normalised data allows us to create subsets where 80% of the array is used for training and 20% is used for testing.
This data is then returned by the `preprocess` method.
</details>

<details>
    <summary>model.py</summary>

This file contains the `LogisticRegression` class which implements a binary classifier using Gradient Descent. It utilises L2 Regularisation to manage the bias-variance tradeoff and numerical clipping to ensure computational stability. 
The implementation follows the standard Scikit-Learn fit/predict pattern. 
The main execution block is utilised to demonstrate a full lifecycle: 
1. Loading and preprocessing the Raisin dataset
2. Training the model 
3. Evaluating performance and logging metrics via the Evaluation class
4. Saving the model to the `/models` directory
</details>


<details>
    <summary>evaluation.py</summary>

The `Evaluation` class calculates binary classification performance metrics by comparing model predictions against ground truth labels. 
Upon initialization, it validates the input data for consistency and computes the fundamental components of a confusion matrix.
This includes: 
* Accuracy: Proportion of all classifications that were correct
* Precision: Proportion of all the model's positive classifications that are actually positive
* Recall value: Proportion of all actual positives that were classified correctly as positives
* F1 score: Evaluates how well a classification model performs on a dataset

This class also throws exceptions if there are issues with the type, shape, or values; logging is also used to give the engineers better tracking.
</details>

<details>
    <summary>tracking.py</summary>

The `ExperimentTracker` class is the MLOps layer of the project. 
The ExperimentTracker ensures that every training run is documented by saving the following to a folder:

* Hyperparameters: Learning rate, epochs, lambda regularization, and threshold used
* Metrics: The accuracy, precision, recall value, and F1 score calculated by the Evaluation class
* Model path: The path to the model which was used
* Metadata: The run_id which is a UUID and a timestamp
* Seed: Seed used to ensure reproducibility

</details>

<details>
    <summary>api.py</summary>

This file sets up the API of the application by leveraging FastAPI. The method `get_model_resource` is used to get the latest model or get a selected model. 
There are also a number of endpoints available:
* GET `/health`
  * `/health` - Returns a status of ok
  * `/model/info` - Returns a list of the weights, bias, and normalised mean & standard deviation used for each feature for a specific model, if no seed is input it will use the latest model that has been trained
  * `/models` - Returns a list of all models
  * `/explain` - Returns a ranked list of the features used from most important to least important, a seed can be input to choose a model but if left empty it will default to the latest model
  * `/experiments/best` - Returns the experiment information which performed best in one of the following chosen metrics: accuracy, precision, recall, f1
* POST
  * `/train` - Allows the user to train the model and input a seed, if no seed is input it will default to 42
  * `/predict` - Takes in float values for the features and allows the user to select a seed that has already been run
</details>





