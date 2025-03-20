from pydantic import BaseModel


class NlpContext(BaseModel):
    device: str = "cuda:0"
    unk_idx: int = 0
    bos_idx: int = 1
    eos_idx: int = 2
    padding_idx: int | None = 3
    min_seq_len: int = 5
    max_seq_len: int = 256
    min_frequency: int = 5
    max_tokens: int | None = None
    unk_str: str = "<unk>"
    bos_str: str = "<bos>"
    eos_str: str = "<eos>"
    pad_str: str = "<pad>"
    special_tokens: list[str] = [unk_str, bos_str, eos_str, pad_str]
