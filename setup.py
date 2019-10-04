import os
import os.path
import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

scripts = []
for n in os.listdir('bin'):
    name = os.path.join('bin', n)
    if os.path.isfile(name) and os.access(name, os.X_OK):
        scripts.append(name)

setuptools.setup(
    name="delta",
    version="0.0.1",
    author="NASA Ames",
    author_email="todo@todo",
    description="Deep learning for satellite imagery",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nasa/delta",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: TODO",
        "Operating System :: OS Independent"
    ],
    install_requires=[
        'psutil',
        'usgs',
        'numpy',
        'scipy',
        'matplotlib',
        'tensorflow==1.12',
        'mlflow',
        'portalocker'
    ],
    scripts=scripts,
    python_requires='>=3.5',
)
