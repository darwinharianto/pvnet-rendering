from setuptools import setup, find_packages
import pvnet_rendering

packages = find_packages(
        where='.',
        include=['pvnet_rendering*']
)

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='pvnet_rendering',
    version=pvnet_rendering.__version__,
    description='Fork of pvnet-rendering',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/darwinharianto/pvnet_rendering",
    author='Clayton Mork',
    author_email='mork.clayton3@gmail.com',
    license='MIT License',
    packages=packages,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'pylint==2.4.4',
        'easydict',
        'lmdb',
        'plyfile',
        'transforms3d',
        'scikit-image',
        'pyclay-annotation_utils'
    ],
    python_requires='>=3.7'
)