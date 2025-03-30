import open3d as o3d
from open3d.web_visualizer import draw as open3d_drawer


def plot_mesh(
    mesh,
    color: tuple[float, float, float] | None = None,
    embed_notebook: bool = False,
):
    mesh.compute_vertex_normals()
    if color is not None:
        mesh.paint_uniform_color(color)

    if embed_notebook:
        open3d_drawer(mesh)
    else:
        o3d.visualization.draw_geometries([mesh])
