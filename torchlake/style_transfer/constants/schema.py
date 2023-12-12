from pydantic import BaseModel


class StyleTransferContext(BaseModel):
    content_layer_names: list[str] = []
    style_layer_names: list[str] = []
