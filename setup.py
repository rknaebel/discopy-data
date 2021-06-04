from setuptools import setup, find_packages

with open("README.md", "r") as fh:
      long_description = fh.read()

setup(name='discopy-data-rknaebel',
      version='0.0.1',
      description='Data and Structures for Neural Discourse Parsing',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='http://github.com/rknaebel/discopy-data',
      author='Rene Knaebel',
      author_email='rknaebel@uni-potsdam.de',
      license='MIT',
      packages=find_packages(),
      zip_safe=False,
      )
