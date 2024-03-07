from setuptools import setup, find_packages

with open('requirements.txt', 'r') as f:
    requirements = f.readlines()

setup(
    name='SyNBEATS',
    version='1.0',
    packages=find_packages('SyNBEATS'),
    package_dir={'': 'src'},
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=requirements,
    python_requires='>=3.7',
)