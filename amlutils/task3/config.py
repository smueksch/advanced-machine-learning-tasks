import json
from dataclasses import dataclass


@dataclass
class Config:
    seed: int
    debug: bool

    disable_comet: bool
    experiment_name: str
    experiment_tags: list[str]

    data_dir: str
    training_set_file: str

    learning_rate: float
    epochs: int
    batch_size: int

    bounding_box_importance: float

    model_checkpoint_dir: str
    model_filename: str


def read_config(config_file: str) -> Config:
    with open(config_file) as file:
        data = json.load(file)
        return Config(**data)


if __name__ == '__main__':
    config = read_config('test.json')
    print(config)
    print(f'Seed: {config.seed} (type {type(config.seed)})')
    print(f'Debug: {config.debug} (type {type(config.debug)})')
    print(
        f'Learning rate: {config.learning_rate} (type {type(config.learning_rate)})')
