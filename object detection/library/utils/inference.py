import torch


def model_predict(model: torch.nn.Module, img: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        if img.dim() == 3:
            img = img.unsqueeze(0)
        pred = model(img)

        pred = pred.detach().cpu()
        img = img.detach().cpu()

    return pred


def generate_grid(grid_x: int, grid_y: int) -> torch.Tensor:
    x_offset, y_offset = torch.meshgrid(
        torch.arange(grid_x),
        torch.arange(grid_y),
        indexing="xy",
    )

    return x_offset, y_offset
