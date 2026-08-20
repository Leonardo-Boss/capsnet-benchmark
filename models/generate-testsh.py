from pathlib import Path
from utils.tools import read_yaml
from os import listdir

augmentations = (
    'none',
    'strong',
)
unseens = (
    None,
    '',
    'large_rotation',
)
config_paths = listdir('configs')

command = '#!/bin/bash\n'
c_start = 'python test.py'
for config_path in config_paths:
    config = read_yaml(Path('configs') / config_path)
    model_p = Path('saved') / 'models' / config['main']['name'] / config_path.removesuffix('.yaml') / 'model_best.pth'
    c_config_model = f"{c_start} -c {Path('configs') / config_path} --model {model_p}"
    for augmentation in augmentations:
        c_aug = f"{c_config_model} --augmentation {augmentation}"
        for unseen in unseens:
            if unseen == None:
                c_unseen = c_config_model
            elif unseen == '':
                c_unseen = f"{c_aug} --unseen-transformation"
            else:
                c_unseen = f"{c_aug} --unseen-transformation {unseen}"
            command += f"{c_unseen}\n"

with open('test.sh', 'w') as f:
    f.write(command)
