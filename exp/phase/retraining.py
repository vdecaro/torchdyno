import os
from ray import tune
from typing import Literal

from ..utils.early_stopping import TrialNoImprovementStopper
from ..trainable import ESNTrainable


def run(dataset: str,
        perc: int,
        mode: Literal['vanilla', 'intrinsic_plasticity'],
        gt: float):
    exp_dir = f"experiments/{dataset}_{perc}_{mode}"
    os.makedirs(exp_dir, exist_ok=True)

    config = get_config(dataset, perc, mode)
    if mode == 'intrinsic_plasticity':
        stopper = TrialNoImprovementStopper(metric='eval_score', 
                                            mode='max', 
                                            patience_threshold=config['PATIENCE'])
    if mode == 'vanilla':
        stopper = lambda trial_id, result: True
    
    reporter = tune.CLIReporter(metric_columns={
                                    'training_iteration': '#Iter',
                                    'eval_score': 'VL-Score', 
                                },
                                parameter_columns={'SIGMA': 'SIGMA', 'HIDDEN_SIZE': '#H', 'LEAKAGE': 'alpha'},
                                infer_limit=3,
                                metric='eval_score',
                                mode='max')

    return tune.run(
        ESNTrainable,
        name=f"retraining",
        stop=stopper,
        local_dir=exp_dir,
        config=config,
        num_samples=3,
        resources_per_trial={"cpu": 1, "gpu": gt},
        keep_checkpoints_num=1,
        checkpoint_score_attr='eval_score',
        checkpoint_freq=1,
        max_failures=5,
        progress_reporter=reporter,
        verbose=1,
        reuse_actors=True
    )


def get_config(name, perc, mode):
    tune_exp = tune.ExperimentAnalysis(f"experiments/{name}_{perc}_{mode}/model_selection", default_metric='eval_score', default_mode='max')
    config = tune_exp.get_best_config()
    config['MODE'] = mode
    return config