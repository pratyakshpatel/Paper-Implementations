rotation forest with a twist

this folder contains a clean and reproducible experiment that compares three tree-based ensemble methods on synthetic classification data:

- random forest (baseline from scikit-learn)
- rotation forest (pca-based feature rotations)
- random rotation forest (random orthogonal rotations)

the objective is to isolate the effect of feature rotations on decision trees. standard trees split along coordinate axes, which can be limiting when the true decision boundary is oblique. rotating the feature space changes the axis alignment seen by each tree and can improve ensemble diversity and predictive behavior.

what this project implements

the main script is [rotation_forest_experiment.py](rotation_forest_experiment.py).

it includes:

1. reproducible synthetic dataset generation with correlated features
2. baseline random forest training and evaluation
3. custom `RotationForestClassifier` implementation
4. custom `RandomRotationForestClassifier` implementation
5. side-by-side evaluation of model accuracy and training time
6. optional tree correlation analysis for ensemble diversity
7. optional 2d decision-boundary plots for visual intuition

rotation forest in plain words

for each tree in the ensemble:

1. split the feature indices into random groups
2. for each group, sample rows from the training set
3. run pca on that feature group
4. place pca components into a block of a full rotation matrix
5. rotate all training features with that matrix
6. train a decision tree on rotated features

because each tree gets a different rotation matrix, the ensemble tends to be both accurate and diverse.

random rotation forest in plain words

this variant keeps the same ensemble design but replaces pca with a random orthogonal transform:

- draw a random matrix
- apply qr decomposition
- use the orthogonal factor as the rotation

this is simpler than pca-based rotation and does not use data geometry directly. it is useful as a comparison point to understand whether pca structure adds value beyond just rotating coordinates.

reproducibility notes

- global seed: `42`
- model seeds are fixed
- dataset generation uses fixed seeds
- train/test split uses fixed seed and stratification

requirements

python packages used:

- numpy
- scipy
- scikit-learn
- matplotlib

run

from this folder, run:

`python rotation_forest_experiment.py`

to skip plotting (useful on headless machines):

`python rotation_forest_experiment.py --no-plot`

expected output

the script prints a comparison table with:

- accuracy
- training time (seconds)
- mean pairwise tree correlation (optional diversity signal)

if plotting is enabled, it also opens a figure with three 2d decision-boundary panels, one for each model.

file structure

- [Readme.md](Readme.md): this narrative description
- [rotation_forest_experiment.py](rotation_forest_experiment.py): complete implementation and experiment runner

implementation design choices

- code follows a scikit-learn-like API with `fit`, `predict`, and `predict_proba`
- feature scaling is applied before rotation for numerical stability
- pca rotations are built block-wise from random feature partitions
- random rotations are generated with qr and sign correction for deterministic behavior under fixed seeds

why this is a useful baseline experiment

this setup gives a controlled way to compare:

- a strong practical baseline (`RandomForestClassifier`)
- a data-driven rotation strategy (`RotationForestClassifier`)
- a non-data-driven rotation strategy (`RandomRotationForestClassifier`)

the result is a clear reference implementation suitable for further research extensions.

