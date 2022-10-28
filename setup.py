"""A setuptools based setup module.

See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    name="nulling",

    version="0.1", 
    
    description="Pulsar Nulling using Mixture models",
    

    url="https://github.com/AkashA98/pulsar_nulling", 
    
    author="Akash Anumarlapudi", 

    author_email="aakash@uwm.edu",
  
    package_dir={"": "src"},
    
    packages=find_packages(where="src"),
   
    python_requires=">=3.8",
    
    install_requires=["numpy", "scipy", "astropy", "matplotlib", "emcee", "scikit-learn", "corner", "tqdm"]
)
