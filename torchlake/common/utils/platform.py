import os
import platform


def get_num_workers() -> int:
    if platform.system() == "Windows":
        return 0
    else:
        return os.cpu_count()
