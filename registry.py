"""
The model registry: making "the ensemble" a single registered, versioned thing.

An ensemble is five .keras files plus a scaler plus a binarizer, which is a
directory, not a model. MLflow's registry wants one model, so this wraps the
whole ensemble - including the scaler and the 0.27 threshold - in a pyfunc.
That way a registered version is self-contained: it takes 90 raw features and
returns tag probabilities, with no way to accidentally pair it with the wrong
scaler.

Each version ALSO logs the plain files under the `ensemble/` artifact path. That
is mild duplication (~1.6 MB) bought deliberately: scoring the champion means
handing a directory to scoring.score_on_holdout, and downloading a known
artifact path is far less fragile than reaching into pyfunc's internal layout.

The champion is tracked by the `champion` alias rather than a stage, because
stages are deprecated in MLflow 3 and aliases are what the SQLite-backed
registry supports.
"""
import os
import shutil
import tempfile

from tracking import CHAMPION_ALIAS, REGISTERED_MODEL_NAME, setup, tracking_uri

ENSEMBLE_ARTIFACT_PATH = 'ensemble'
PYFUNC_ARTIFACT_PATH = 'model'
MACRO_F1_METRIC = 'macro_f1'


class EnsembleTagger:
    """
    pyfunc wrapper. Defined as a plain class and adapted below so this module can
    be imported (and unit-tested) without mlflow present.
    """

    def load_context(self, context):
        import pickle

        os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
        import tensorflow as tf

        with open(context.artifacts['scaler'], 'rb') as f:
            self.scaler = pickle.load(f)
        with open(context.artifacts['binarizer'], 'rb') as f:
            self.binarizer = pickle.load(f)

        self.models = []
        i = 1
        while 'model_%d' % i in context.artifacts:
            self.models.append(
                tf.keras.models.load_model(context.artifacts['model_%d' % i], compile=False))
            i += 1

    def predict(self, context, model_input, params=None):
        """90 unscaled features in, one probability per tag out."""
        import numpy as np
        import pandas as pd

        X = np.asarray(model_input, dtype=float)
        X_scaled = self.scaler.transform(X)
        probs = np.mean([m.predict(X_scaled, verbose=0) for m in self.models], axis=0)
        return pd.DataFrame(probs, columns=list(self.binarizer.classes_))


def _pyfunc_class():
    """Build the real mlflow.pyfunc.PythonModel subclass at call time."""
    import mlflow.pyfunc

    return type('EnsembleTaggerModel', (mlflow.pyfunc.PythonModel,), {
        'load_context': EnsembleTagger.load_context,
        'predict': EnsembleTagger.predict,
    })


def _artifact_map(model_dir, num_models):
    from scoring import BINARIZER_NAME, MODEL_GLOB, SCALER_NAME

    artifacts = {
        'scaler': os.path.join(model_dir, SCALER_NAME),
        'binarizer': os.path.join(model_dir, BINARIZER_NAME),
    }
    for i in range(1, num_models + 1):
        artifacts['model_%d' % i] = os.path.join(model_dir, MODEL_GLOB % i)
    return artifacts


def register_ensemble(model_dir, summary, per_tag, params=None, tags=None,
                      run_name=None, register=True):
    """
    Log one ensemble as an MLflow run and, optionally, a new registered version.

    Returns (run_id, version) - version is None when register=False.
    """
    from scoring import count_models

    # setup() returns the configured mlflow module - taking it from there rather
    # than importing separately guarantees the tracking URI and experiment are
    # set before anything is logged.
    mlflow = setup()
    num_models = count_models(model_dir)
    if num_models == 0:
        raise FileNotFoundError("No models found in %s" % model_dir)

    with mlflow.start_run(run_name=run_name) as run:
        if params:
            mlflow.log_params(params)
        if tags:
            mlflow.set_tags(tags)

        mlflow.log_metrics({k: v for k, v in summary.items()
                            if isinstance(v, (int, float))})
        mlflow.set_tags({k: str(v) for k, v in summary.items()
                         if not isinstance(v, (int, float))})

        if per_tag is not None:
            staging = os.path.join('.cache', 'mlflow_staging')
            os.makedirs(staging, exist_ok=True)
            csv_path = os.path.join(staging, 'per_tag_report.csv')
            per_tag.to_csv(csv_path, index=False)
            mlflow.log_artifact(csv_path)
            for _, row in per_tag.iterrows():
                key = 'f1_tag_' + str(row['tag']).replace(' ', '_').replace('/', '_')
                mlflow.log_metric(key, float(row['f1']))

        # Plain files, for scoring a champion later.
        for name, path in _artifact_map(model_dir, num_models).items():
            mlflow.log_artifact(path, artifact_path=ENSEMBLE_ARTIFACT_PATH)

        manifest = os.path.join(model_dir, 'split_manifest.json')
        if os.path.exists(manifest):
            mlflow.log_artifact(manifest)

        config = os.path.join(model_dir, 'model_config.json')
        if os.path.exists(config):
            mlflow.log_artifact(config)

        mlflow.pyfunc.log_model(
            name=PYFUNC_ARTIFACT_PATH,
            python_model=_pyfunc_class()(),
            artifacts=_artifact_map(model_dir, num_models),
            registered_model_name=REGISTERED_MODEL_NAME if register else None)

        run_id = run.info.run_id

    version = None
    if register:
        version = _latest_version_for_run(run_id)
    return run_id, version


def _client():
    import mlflow

    mlflow.set_tracking_uri(tracking_uri())
    return mlflow.tracking.MlflowClient()


def _latest_version_for_run(run_id):
    client = _client()
    versions = client.search_model_versions("name='%s'" % REGISTERED_MODEL_NAME)
    for v in versions:
        if v.run_id == run_id:
            return int(v.version)
    return None


def get_champion():
    """
    (version, run_id, macro_f1) for the current champion, or None if unset.

    Returns None rather than raising when the registered model does not exist at
    all - that is the legitimate "first ever model" case the gate must handle.
    """
    client = _client()
    try:
        mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)
    except Exception:
        return None
    if mv is None:
        return None

    run = client.get_run(mv.run_id)
    f1 = run.data.metrics.get(MACRO_F1_METRIC)
    return int(mv.version), mv.run_id, (float(f1) if f1 is not None else None)


def get_best_ever(metric=MACRO_F1_METRIC):
    """
    Best metric value across every registered version.

    This is the ratchet guard's floor: without it, a chain of candidates each
    barely clearing the incumbent walks the model steadily downhill.
    """
    client = _client()
    try:
        versions = client.search_model_versions("name='%s'" % REGISTERED_MODEL_NAME)
    except Exception:
        return None

    best = None
    for v in versions:
        try:
            run = client.get_run(v.run_id)
        except Exception:
            continue
        value = run.data.metrics.get(metric)
        if value is not None and (best is None or value > best):
            best = float(value)
    return best


def set_champion(version):
    """Point the `champion` alias at a version."""
    client = _client()
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS, str(version))


def download_champion_ensemble(dest=None):
    """
    Fetch the champion's raw ensemble files so it can be re-scored on the
    current fixed split. Re-scoring rather than trusting the stored metric is
    what keeps the comparison honest when the dataset or split has moved.
    """
    import mlflow

    champ = get_champion()
    if champ is None:
        return None
    _version, run_id, _f1 = champ

    mlflow.set_tracking_uri(tracking_uri())
    dest = dest or tempfile.mkdtemp(prefix='champion_')
    local = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=ENSEMBLE_ARTIFACT_PATH, dst_path=dest)
    return local


def cleanup(path):
    if path and os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
