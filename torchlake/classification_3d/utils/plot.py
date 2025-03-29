from pathlib import Path
import open3d as o3d


def plot_mesh(
    path: Path | str,
    color: tuple[float, float, float] | None = None,
):
    mesh = o3d.io.read_triangle_mesh(Path(path))
    mesh.compute_vertex_normals()
    if color is not None:
        mesh.paint_uniform_color(color)

    o3d.visualization.draw_geometries([mesh])
