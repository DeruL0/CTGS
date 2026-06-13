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
                "boundary_field.cu",
                "bulk_intensity_query.cu",
                "local_query.cu",
                "query_density.cu",
                "render_slice_patch.cu",
                "surface_regularization.cpp",
                "uniform_grid.cu",
            ],
            extra_compile_args={
                "cxx": cxx_compiler_flags,
                "nvcc": ["-O2"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
