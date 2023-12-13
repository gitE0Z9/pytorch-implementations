import torch
import torch.nn.functional as F
from torchvision import transforms

from ..models.neural_doodle.loss import MrfLoss, get_multiscale_sizes
from ..models.neural_doodle.model import AuxiliaryNetwork
from ..models.neural_style_transfer.loss import NeuralStyleTransferLoss
from ..models.neural_style_transfer.model import NeuralStyleTransfer


def run_neural_style_transfer(
    model: NeuralStyleTransfer,
    criterion: NeuralStyleTransferLoss,
    content: torch.Tensor,
    style: torch.Tensor,
    num_steps: int = 300,
    save_iter: int = 50,
):
    """Run the style transfer."""
    output = content.clone().contiguous()

    optimizer = torch.optim.LBFGS([output.requires_grad_()])

    step = 0
    while step <= num_steps:

        def closure():
            optimizer.zero_grad()

            # correct the values of updated input image
            output.data.clamp_(0, 1)

            output_features = model(output, "style")
            content_feature = model(content, "content")
            style_features = model(style, "style")
            loss, content_score, style_score = criterion(
                content_feature[0], style_features, output_features
            )
            loss.backward()

            nonlocal step
            step += 1
            if step % save_iter == 0:
                print(f"run {step}:")
                print(
                    f"Total Loss: {loss.item():4f}",
                    f"Content Loss: {content_score.item():4f}",
                    f"Style Loss : {style_score.item():4f}",
                )
                print()

            return loss

        optimizer.step(closure)

    output.data.clamp_(0, 1)

    return output


def run_neural_doodle(
    model: AuxiliaryNetwork,
    criterion: MrfLoss,
    style: torch.Tensor,
    style_mask: torch.Tensor,
    input_mask: torch.Tensor,
    max_scale: float = 1,
    num_steps: int = 300,
    save_iter: int = 50,
):
    """Run the style transfer."""

    _, _, h, w = style.shape
    hw_list = get_multiscale_sizes(h, w, max_scale=max_scale)
    hw_list.reverse()
    print("scales:", hw_list)

    output = torch.rand(1, 3, *hw_list[0]).to(style.device)

    # calculate multiscale
    for phase_number, size in enumerate(hw_list):
        print(f"phases {phase_number+1} begins")

        _style = transforms.Resize(size, antialias=True)(style)
        _style_mask = transforms.Resize(size, antialias=True)(style_mask)
        _input_mask = transforms.Resize(size, antialias=True)(input_mask)

        optimizer = torch.optim.LBFGS([output.requires_grad_()])

        for step in range(num_steps + 1):
            output.data.clamp_(0, 1)
            optimizer.zero_grad()

            with torch.no_grad():
                style_features = model(_style, _style_mask)
            input_features = model(output, _input_mask)

            loss, _, style_score, _ = criterion(style_features, input_features)
            loss.backward()

            optimizer.step(lambda: loss)

            if (step % save_iter) == 0:
                print(
                    f"{step:<6}",
                    f"Total Loss: {loss.item():4f}",
                    # f"Content Loss: {content_score.item():4f}",
                    f"Style Loss : {style_score.item():4f}",
                    # f"Smooth Loss : {tv_score.item():4f}",
                )

        output = F.interpolate(output, scale_factor=2).detach()

    output.data.clamp_(0, 1)

    return output
