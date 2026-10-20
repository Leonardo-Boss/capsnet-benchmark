"""Generates test.sh: every config x checkpoint-selection x test condition.

Checkpoints written by the (A1-patched) trainer:
    model_best.pth      -> selected on clean, deterministic validation  ("clean")
    model_best_aug.pth  -> selected on fixed-seed augmented validation   ("aug")

Results go to <output-root>/<selection>/, so the two selections never mix and
never collide with CSVs from older runs (test.py silently SKIPS conditions whose
CSV already exists).

    python generate-testsh.py                    # both selections
    python generate-testsh.py --selection clean --only cifar_10
"""
import argparse
from os import listdir
from pathlib import Path

from utils.tools import read_yaml

ap = argparse.ArgumentParser()
ap.add_argument('--selection', choices=['clean', 'aug', 'both'], default='both')
ap.add_argument('--output-root', default='results_v2')
ap.add_argument('--only', default=None, help='substring filter on the config file name')
a = ap.parse_args()

CKPT = {'clean': 'model_best.pth', 'aug': 'model_best_aug.pth'}
selections = list(CKPT) if a.selection == 'both' else [a.selection]

augmentations = (
    'none',
    'strong'
)
unseens = (
    None,
    '',
    'large_rotation'
)
config_paths = sorted(listdir('configs'))

command = '#!/bin/bash\nset -e\n'
c_start = 'python test.py'
for config_path in config_paths:
    if a.only and a.only not in config_path:
        continue
    config = read_yaml(Path('configs') / config_path)
    # same location train.py wrote to: <save_dir>/models/<name>/<run_id>/
    run_dir = (Path(config['trainer']['save_dir']) / 'models' / config['main']['name']
               / config_path.removesuffix('.yaml'))
    for selection in selections:
        model_p = run_dir / CKPT[selection]
        c_config_model = (f"{c_start} -c {Path('configs') / config_path} --model {model_p} "
                          f"--output_dir {Path(a.output_root) / selection}")
        for augmentation in augmentations:
            c_aug = f"{c_config_model} --augmentation {augmentation}"
            for unseen in unseens:
                if unseen is None:
                    c_unseen = c_aug
                elif unseen == '':
                    c_unseen = f"{c_aug} --unseen-transformation"
                else:
                    c_unseen = f"{c_aug} --unseen-transformation {unseen}"
                command += f"{c_unseen}\n"

with open('test.sh', 'w') as f:
    f.write(command)
print(f"wrote test.sh ({command.count('python test.py')} evaluations)")