from setuptools import find_packages, setup

setup(
    name='ergo',
    version='0.0.1',
    packages=find_packages(
        where='.',
        include=['ergo*'],
    ),
    install_requires=[
        'numpy == 1.23.1',
        'scikit-learn == 1.1.1'
    ]
)
