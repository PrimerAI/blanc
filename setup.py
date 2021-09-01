import setuptools

from blanc.__version__ import __version__

with open("README.md", encoding="utf-8") as reader:
    long_description = reader.read()

with open("requirements.txt") as reader:
    requirements = [line.strip() for line in reader]

setuptools.setup(
    name="blanc",
    version=__version__,
    author="Primer AI",
    author_email="blanc@primer.ai",
    description="Human-free quality estimation of document summaries",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PrimerAI/blanc",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    entry_points={
        "console_scripts": ["blanc=blanc.__main__:main"],
    },
)
