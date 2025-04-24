from setuptools import setup, find_packages

setup(
    name="h2xh2",
    version="0.0.0-alpha",
    python_requires=">=3.12",
    description="Tools for reproducing the results from the paper",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="Proprietary Licence",
    include_package_data=True,
    packages=find_packages(),
    install_requires=[
        "pytket==1.38.0",
        "pytket-quantinuum[pecos]==0.42.0",
    ],
    classifiers=[
        "Environment :: Console",
        "Programming Language :: Python :: 3.12",
        "License :: Other/Proprietary License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
    ],
    zip_safe=False,
)
