from glob import glob

import click

from .constants.enums import MediaType, NetworkStage
from .controller.predictor import Predictor


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
    "-f",
    "--files",
    "files",
    type=str,
    help="glob pattern to image or video",
    required=True,
)  # TODO: glob parse error
@click.option(
    "-media",
    "--media",
    "media_type",
    type=str,
    help="image or video",
    required=True,
)
@click.option(
    "-w",
    "--weight",
    "weight",
    type=str,
    help="path to weight",
    required=True,
)
@click.option(
    "-show",
    "--show",
    "show",
    is_flag=True,
    help="whether to show the result or not",
)
@click.option(
    "--save-dir",
    "save_dir",
    type=str,
    help="where to save the result",
)
def main(
    config: str,
    network_type: str,
    dataset: str,
    files: str,
    media_type: str,
    weight: str,
    show: bool,
    save_dir: str,
):
    # media type warning
    warning_message = f"Only {','.join(item.value for item in MediaType)} supported."
    assert media_type.upper() in MediaType.__members__, warning_message
    media_path = glob(files)

    controller = Predictor(config, dataset, network_type, NetworkStage.INFERENCE.value)

    media_type = media_type.lower()
    if media_type == MediaType.IMAGE.value:
        assert len(media_path) > 0, "Images are not found."
        controller.predict_image_file(weight, media_path, show, save_dir)
    elif media_type == MediaType.VIDEO.value:
        print("Number of videos: ", len(media_path))
        assert len(media_path) == 1, "Only one video supported."
        controller.predict_video_file(weight, media_path[0], show, save_dir)

    print("Prediction done!")


if __name__ == "__main__":
    main()
