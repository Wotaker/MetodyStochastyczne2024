from setuptools import setup, find_packages

setup(
    name='gpr',
    version='0.0.1',
    packages=find_packages(include=[
        'source_code',
        'source_code.*'
    ]),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'pyhms' # Unavailable on PyPI yet. Install from source: pip install git+https://github.com/agh-a2s/pyhms.git@main
    ],
)