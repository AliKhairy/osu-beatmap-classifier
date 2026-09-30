"""
MLflow wiring: where runs go and what a run is required to carry.

Backend is a local SQLite file plus a local artifact directory, both gitignored
and dockerignored. SQLite rather than the default file store because the model
registry - specifically alias support, which is how `champion` is tracked - only
works against a database-backed store.

Everything here is import-light at call time: cli.py imports this module inside
the subcommand that needs it, so `cli.py --help` never pays for mlflow, and the
CLI keeps working in an environment where mlflow is not installed at all.
"""
import os

DEFAULT_TRACKING_URI = 'sqlite:///mlflow.db'
DEFAULT_ARTIFACT_DIR = 'mlruns'
EXPERIMENT_NAME = 'osu-beatmap-classifier'
REGISTERED_MODEL_NAME = 'osu-tagger'
CHAMPION_ALIAS = 'champion'


def tracking_uri():
    """
    Resolve where runs are recorded.

    MLFLOW_TRACKING_URI is honoured so verification runs can point at a
    throwaway database and leave the real one alone - that is how the promotion
    tests avoid touching the live registry.
    """
    return os.environ.get('MLFLOW_TRACKING_URI', DEFAULT_TRACKING_URI)


def setup(experiment_name=EXPERIMENT_NAME):
    """Point mlflow at the local backend and make sure the experiment exists."""
    import mlflow

    uri = tracking_uri()
    mlflow.set_tracking_uri(uri)

    artifact_root = os.path.abspath(DEFAULT_ARTIFACT_DIR)
    os.makedirs(artifact_root, exist_ok=True)

    existing = mlflow.get_experiment_by_name(experiment_name)
    if existing is None:
        # pathlib-free file URI: mlflow wants a URI, and a bare Windows path
        # like C:\... is parsed as a scheme named "c".
        mlflow.create_experiment(
            experiment_name,
            artifact_location='file:///' + artifact_root.replace('\\', '/'))
    mlflow.set_experiment(experiment_name)
    return mlflow


def log_evaluation(summary, per_tag, params=None, extra_artifacts=None,
                   run_name=None, tags=None, nested=False):
    """
    Record one evaluation as an MLflow run.

    Logs per-tag F1 as individual metrics as well as shipping the CSV, because
    the CSV is for a human reading one run and the metrics are for comparing
    sixty runs in the UI without opening any of them.
    """
    # setup(), not just set_tracking_uri(). Setting the URI alone leaves the
    # ACTIVE EXPERIMENT at whatever it happened to be - which, in a fresh
    # process, is "Default" (id 0). Runs from evaluate --holdout, drift and the
    # flow's evaluate task were all landing there instead of alongside the
    # registered models, so the UI showed one experiment with a single run and
    # the rest scattered in Default.
    mlflow = setup()

    with mlflow.start_run(run_name=run_name, nested=nested) as run:
        if params:
            mlflow.log_params(params)
        if tags:
            mlflow.set_tags(tags)

        scalar = {k: v for k, v in summary.items() if isinstance(v, (int, float))}
        mlflow.log_metrics(scalar)

        text_fields = {k: str(v) for k, v in summary.items()
                       if not isinstance(v, (int, float))}
        if text_fields:
            mlflow.set_tags(text_fields)

        if per_tag is not None:
            for _, row in per_tag.iterrows():
                # Tag names contain spaces and hyphens; mlflow metric keys allow
                # both, but not every character, so normalise defensively.
                key = 'f1_tag_' + str(row['tag']).replace(' ', '_').replace('/', '_')
                mlflow.log_metric(key, float(row['f1']))

            csv_path = os.path.join(_run_tmp(), 'per_tag_report.csv')
            per_tag.to_csv(csv_path, index=False)
            mlflow.log_artifact(csv_path)

        for path in (extra_artifacts or []):
            if path and os.path.exists(path):
                mlflow.log_artifact(path)

        return run.info.run_id


def _run_tmp():
    d = os.path.join('.cache', 'mlflow_staging')
    os.makedirs(d, exist_ok=True)
    return d


def describe_run(run_id):
    """Read a run back out of the store - used to verify what was logged."""
    import mlflow

    mlflow.set_tracking_uri(tracking_uri())
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    return {
        'run_id': run_id,
        'status': run.info.status,
        'params': dict(run.data.params),
        'metrics': dict(run.data.metrics),
        'tags': {k: v for k, v in run.data.tags.items() if not k.startswith('mlflow.')},
        'artifacts': [f.path for f in client.list_artifacts(run_id)],
    }
