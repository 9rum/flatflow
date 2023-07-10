from setuptools import setup, find_packages

from chronica import __version__


if __name__ == "__main__":
  setup(
    name="chronica",
    version=__version__,
    description="A data-imbalance-aware scheduler for distributed deep learning",
    long_description=open("README.rst").read(),
    long_description_content_type="text/x-rst",
    author="Sogang University",
    url="",
    download_url="",
    packages=find_packages(exclude=("tests")),
    license="Apache-2.0",
    keywords="distributed deep learning, straggler, scheduler, data imbalance",
  )
