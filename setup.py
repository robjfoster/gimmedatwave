from setuptools import setup, find_packages, find_namespace_packages

setup(
    name='gimmedatwave',
    version='0.1',
    packages=find_namespace_packages(include=['gimmedatwave.*']),
)
