import click
from controller.trainer import Trainer
from constants.enums import NetworkStage

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
    "--stage",
    "stage",
    type=str,
    help="for classifier, FINETUNE, SCRATCH, INFERENCE supported",
    required=True,
    default=NetworkStage.SCRATCH.value,
)
@click.option(
    "-desc",
    "--description",
    "description",
    type=str,
    help="describe this training",
    default="",
)
def main(config: str, network_type: str, dataset: str, stage: str, description: str):
    control = Trainer(config, dataset, network_type, stage)
    control.train(description)
    print("Training done!")


if __name__ == "__main__":
    main()
