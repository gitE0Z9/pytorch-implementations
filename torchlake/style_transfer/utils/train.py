import torch
import torch.nn.functional as F
from torchvision import transforms

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
