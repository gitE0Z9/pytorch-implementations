import torch
import torch.nn.functional as F
import torchvision.transforms as T

from ..models.neural_doodle import MRFLoss, NeuralDoodle


def get_multiscale_sizes(
    h: int,
    w: int,
    max_scale: float = 0.5,
    min_size: int = 64,
) -> list[int]:
    max_h, max_w = int(h * max_scale), int(w * max_scale)
    hw_list = [[max_h, max_w]]
    while 1:
        new_h, new_w = max_h // 2, max_w // 2
        if new_h >= min_size or new_w >= min_size:
            hw_list.append([new_h, new_w])
            max_h, max_w = new_h, new_w
        else:
            break
    return hw_list


def run_neural_doodle(
    model: NeuralDoodle,
    criterion: MRFLoss,
    style: torch.Tensor,
    style_mask: torch.Tensor,
    input_mask: torch.Tensor,
    max_scale: float = 1,
    num_steps: int = 300,
    save_iter: int = 50,
):
    """Run the nerual doodle."""
    _, _, h, w = style.shape
    hw_list = get_multiscale_sizes(h, w, max_scale=max_scale)
    hw_list.reverse()
    print("scales:", hw_list)

    output = torch.rand(1, 3, *hw_list[0]).to(style.device)

    # calculate multiscale
    for phase_number, size in enumerate(hw_list):
        print(f"phases {phase_number+1} begins")

        _style = T.Resize(size, antialias=True)(style)
        _style_mask = T.Resize(size, antialias=True)(style_mask)
        _input_mask = T.Resize(size, antialias=True)(input_mask)

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
