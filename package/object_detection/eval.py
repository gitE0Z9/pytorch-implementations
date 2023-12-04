from glob import glob

import click

from object_detection.constants.enums import NetworkStage
from object_detection.controller.evaluator import Evaluator


@click.command()
@click.option(
    "-c",
    "--config",
    "config",
    type=str,
    help="which config to use",
    required=True,
)
@click.option(
    "-nt",
    "--network-type",
    "network_type",
    type=str,
    help="DETECTOR or CLASSIFIER",
    required=True,
)
@click.option(
    "-d",
    "--dataset",
    "dataset",
    type=str,
    help="which dataset to use",
    required=True,
)
@click.option(
    "-w",
    "--weight",
    "weight",
    type=str,
    help="glob pattern to choose weight",
    required=True,
)
@click.option(
    "-s",
    "--save",
    "save",
    is_flag=True,
    help="whether save mAP or not",
)
@click.option(
    "-desc",
    "--description",
    "description",
    type=str,
    help="describe this evaluation",
    default="",
)
def main(
    config: str,
    network_type: str,
    dataset: str,
    weight: str,
    save: bool,
    description: str,
):
    control = Evaluator(config, dataset, network_type, NetworkStage.INFERENCE.value)
    weight_path = glob(weight)
    control.evaluate(weight_path, save, description)
    print("Evaluation done!")


if __name__ == "__main__":
    main()
