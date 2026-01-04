# Note: This file is kept for compatibility but dependencies are managed in pyproject.toml
# Use `uv sync` to install dependencies

from setuptools import setup, find_packages

setup(
    name='a2',
    version='0.2.0',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    description="Action Prior Alignment - Language-conditioned Pick and Place in Clutter",
    author='Kechun Xu',
    author_email='kcxu@zju.edu.cn',
    maintainer='Denis Grachev',
    url='https://github.com/grach0v/Action-Prior-Alignment',
    # Dependencies are managed in pyproject.toml
)
