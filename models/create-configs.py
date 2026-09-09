from utils.tools import read_yaml, write_yaml

def update_recursive(d, u):
    for k, v in u.items():
        if isinstance(v, dict):
            d.get(k, {}).update(v) if isinstance(d.get(k), dict) else d.__setitem__(k, v)
        else:
            d[k] = v

# config = read_yaml('config.yaml')
# ecaps = read_yaml('config-ecaps.yaml')
# deit = read_yaml('config-deit-tiny.yaml')
# resnet = read_yaml('config-resnet-18.yaml')
config = read_yaml('config-flame.yaml')
ecaps = read_yaml('config-ecaps-flame.yaml')
deit = read_yaml('config-deit-tiny-flame.yaml')
resnet = read_yaml('config-resnet-18-flame.yaml')

models = (ecaps, deit, resnet,)
# models = (ecaps,)
databases = (
        # {'name':'cifar_10', 'type':'Cifar10DataLoader'},
        {'name':'flame', 'type':'FlameDataLoader'},
)
augmentations = (
    'none',
    'standard',
    'strong'
)
data_fractions = (
    0.33,
    0.66,
    1,
)
seeds = (
    1,
    2,
    3,
)
for model in models:
    update_recursive(model, config)
    for database in databases:
        model['main']['name'] = f"{model['main']['name']}_{database['name']}"
        model['data_loader']['type'] = database['type']
        for augmentation in augmentations:
            model['data_loader']['args']['augmentation'] = augmentation
            for data_fraction in data_fractions:
                model['data_loader']['args']['data_fraction'] = data_fraction
                for seed in seeds:
                    model['main']['seed'] = seed
                    write_yaml(model, f"configs/{model['main']['name']}_{augmentation}_{data_fraction}_{seed}.yaml")
