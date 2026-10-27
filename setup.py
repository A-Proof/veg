"""Optional native CPU kernels; NumPy remains a portable fallback."""
from setuptools import Extension, setup

setup(ext_modules=[Extension("veg._native", ["veg/native.c"], optional=True)])
