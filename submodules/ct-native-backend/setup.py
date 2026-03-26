from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os


cxx_compiler_flags = []
if os.name == "nt":
    cxx_compiler_flags.extend(["/wd4624"])


setup(
    name="ct_native_backend",
    packages=["ct_native_backend"],
    ext_modules=[
        CUDAExtension(
            name="ct_native_backend._C",
            sources=[
                "ext.cpp",
                "neighbor_index.cu",
                "plane_ops.cpp",
                "query_density.cu",
                "render_slice_patch.cu",
            ],
            extra_compile_args={
                "cxx": cxx_compiler_flags,
                "nvcc": ["-O2"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
