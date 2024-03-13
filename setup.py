from setuptools import setup, find_packages
# Read the contents of requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()
setup(
    name='amb',
    version='0.1',
    packages=find_packages(),
    author='PhD Mikel Martinez',
    author_email='mikel.martinez@unizar.es',
    description='Library gathering all common scripts for the AMB research team',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/mikelunizar/amb',
    install_requires=[requirements],
    python_requires='>=3.6',
)