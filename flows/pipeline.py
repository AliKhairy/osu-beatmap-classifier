"""
The training pipeline as a Prefect flow: train -> evaluate -> gate -> export.

Tasks call the project's Python functions directly rather than shelling out to
cli.py. Subprocesses would give the flow nothing but an exit code, so a failure
would arrive as "exit 2" with the real traceback stranded in a captured stdout
string. Calling in-process means Prefect sees the actual exception, retries mean
something, and the returned values (metrics, paths) flow between tasks as
objects instead of being re-parsed from text.

The ordering constraint is the point of the whole thing: export_onnx depends on
promote, and promote raises on rejection. A rejected candidate therefore cannot
reach the export step - the gate is structural, not a convention someone has to
remember.

build-dataset is opt-in. It needs network access and API credentials from .env,
and it rewrites the dataset the fixed split is defined over, so it is never
something to run by accident.
"""
import os
from datetime import datetime

from prefect import flow, get_run_logger, task


@task(name='build-dataset')
def build_dataset_task(max_maps, output):
    from dataset_builder import build_full_dataset, save_dataset

    logger = get_run_logger()
    logger.info("Scraping up to %d maps into %s", max_maps, output)
    data = build_full_dataset(max_maps=max_maps, offset=0)
    if not save_dataset(data, output):
        raise RuntimeError(
            "Dataset build produced no maps. Check ECHO_API_TOKEN and the osu! "
            "API credentials.")
    return output


@task(name='train-ensemble')
def train_task(dataset, models, epochs, train_seed, candidate_dir):
    from ensemble_evaluator import train_and_evaluate_ensemble

    logger = get_run_logger()
    logger.info("Training %d model(s) for %d epoch(s) into %s",
                models, epochs, candidate_dir)
    result = train_and_evaluate_ensemble(
        num_models=models, dataset=dataset, epochs=epochs,
        train_seed=train_seed, out_dir=candidate_dir)
    if result is None:
        raise RuntimeError("Training produced no models")
    return result


@task(name='evaluate-holdout')
def evaluate_task(candidate_dir, dataset, threshold):
    import split as split_mod
    from metrics_report import format_summary
    from scoring import score_on_holdout
    from tracking import log_evaluation

    logger = get_run_logger()
    prepared = split_mod.prepare_dataset(dataset)
    sp = split_mod.fixed_split(prepared)
    summary, per_tag, _ = score_on_holdout(candidate_dir, prepared, sp, threshold=threshold)
    logger.info("Candidate on fixed split: %s", format_summary(summary))

    run_id = log_evaluation(
        summary, per_tag,
        params={'split_seed': sp.seed, 'threshold': threshold,
                'dataset_sha256': prepared.dataset_sha, 'split_hash': sp.split_hash,
                'model_dir': candidate_dir},
        run_name='pipeline-evaluate',
        tags={'pipeline_stage': 'evaluate'})
    logger.info("Logged evaluation run %s", run_id)
    return summary['macro_f1']


@task(name='promote')
def promote_task(candidate_dir, dataset, threshold, tolerance, root_dir):
    """
    The gate. Raises on rejection, which is what stops the export task.

    Reuses cli.cmd_promote so the flow and the command line cannot drift apart -
    there is one gate implementation, not two that agree today.
    """
    import argparse

    import cli

    logger = get_run_logger()
    args = argparse.Namespace(
        candidate=candidate_dir, dataset=dataset, threshold=threshold,
        tolerance=tolerance, root_dir=root_dir, run_name='pipeline-promote')

    code = cli.cmd_promote(args)
    if code != 0:
        raise RuntimeError(
            "Promotion gate rejected the candidate (exit %d). Export is skipped: "
            "the models currently in %s stay in place." % (code, root_dir))
    logger.info("Gate passed; candidate promoted into %s", root_dir)
    return root_dir


@task(name='export-onnx')
def export_task(root_dir):
    import argparse

    import cli

    logger = get_run_logger()
    args = argparse.Namespace(model_dir=root_dir, out_dir=root_dir)
    code = cli.cmd_export_onnx(args)
    if code != 0:
        raise RuntimeError("ONNX export failed with exit code %d" % code)

    produced = [f for f in sorted(os.listdir(root_dir))
                if f.endswith('.onnx') or f == 'model_config.json']
    logger.info("Exported: %s", produced)
    return produced


@flow(name='osu-tagger-training')
def training_pipeline(dataset='ml_dataset.json', models=5, epochs=100, train_seed=None,
                      tolerance=None, threshold=0.27, candidate_dir=None, root_dir='.',
                      build_dataset=False, max_maps=5000):
    logger = get_run_logger()

    if candidate_dir is None:
        candidate_dir = os.path.join(
            'candidates', datetime.now().strftime('%Y%m%d-%H%M%S'))

    if build_dataset:
        dataset = build_dataset_task(max_maps, dataset)

    train_task(dataset, models, epochs, train_seed, candidate_dir)
    macro_f1 = evaluate_task(candidate_dir, dataset, threshold)
    logger.info("Candidate macro F1: %.4f", macro_f1)

    promoted_root = promote_task(candidate_dir, dataset, threshold, tolerance, root_dir)
    exported = export_task(promoted_root)

    logger.info("Pipeline complete. Exported %d file(s) to %s", len(exported), promoted_root)
    return {'candidate_dir': candidate_dir, 'macro_f1': macro_f1,
            'root_dir': promoted_root, 'exported': exported}


if __name__ == '__main__':
    training_pipeline()
