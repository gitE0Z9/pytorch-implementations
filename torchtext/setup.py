from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name="torchtext",
    version="0.0.4",
    packages=["torchtext", "torchtext.vocab", "torchtext.data"],
    ext_modules=[
        CppExtension(
            "torchtext.vocab._torchtext",
            [
                "torchtext/vocab/bindings.cpp",
                "torchtext/vocab/vocab.cpp",
                "torchtext/vocab/vocab_factory_cpp.cpp",
                "torchtext/vocab/common.cpp",
            ],
            extra_compile_args=["-std=c++17"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
