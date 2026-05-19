import torch

from ..models.neural_style_transfer import NeuralStyleTransferLoss


def run_neural_style_transfer(
    criterion: NeuralStyleTransferLoss,
    content: torch.Tensor,
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

            loss, content_score, style_score = criterion(output, content)
            loss.backward()

            nonlocal step
            step += 1
            if step % save_iter == 0:
                print(
                    f"run {step}:",
                    f"Total Loss: {loss.item():4f}",
                    f"Content Loss: {content_score.item():4f}",
                    f"Style Loss : {style_score.item():4f}",
                    "\n",
                )

            return loss

        optimizer.step(closure)

    output.data.clamp_(0, 1)

    return output
