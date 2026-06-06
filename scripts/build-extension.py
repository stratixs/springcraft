from __future__ import annotations

import os
import platform
import shutil
import sys
from pathlib import Path

import numpy as np
from Cython.Build import cythonize
from setuptools import Distribution, Extension
from setuptools.command.build_ext import build_ext

if sys.platform == "win32":
    COMPILE_ARGS = ["/O2"]
else:
    COMPILE_ARGS = ["-O3", "-march=native"]
    if platform.machine().lower() in ["x86_64", "amd64", "x86"]:
        COMPILE_ARGS.extend(["-msse", "-msse2", "-mfma", "-mfpmath=sse"])
LINK_ARGS = []
INCLUDE_DIRS = [np.get_include()]
LIBRARIES = [] if sys.platform == "win32" else ["m"]


def build() -> None:
    extensions = [
        Extension(
            "*",
            ["src/springcraft/*.pyx"],
            language="c++",
            extra_compile_args=COMPILE_ARGS,
            extra_link_args=LINK_ARGS,
            include_dirs=INCLUDE_DIRS,
            libraries=LIBRARIES,
        )
    ]
    ext_modules = cythonize(
        extensions,
        include_path=INCLUDE_DIRS,
        compiler_directives={"binding": True, "language_level": 3},
    )

    distribution = Distribution({"name": "package", "ext_modules": ext_modules})

    cmd = build_ext(distribution)
    cmd.ensure_finalized()
    cmd.run()

    # Copy built extensions back to the project
    for output in cmd.get_outputs():
        output = Path(output)
        relative_extension = Path("src") / output.relative_to(cmd.build_lib)

        shutil.copyfile(output, relative_extension)
        mode = os.stat(relative_extension).st_mode
        mode |= (mode & 0o444) >> 2
        os.chmod(relative_extension, mode)


if __name__ == "__main__":
    build()
