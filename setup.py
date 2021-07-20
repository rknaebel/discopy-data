from setuptools import setup, find_packages

with open("README.md", "r") as fh:
      long_description = fh.read()

setup(name='discopy-data-rknaebel',
      version='1.0.0',
      description='Data and Structures for Neural Discourse Parsing',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='http://github.com/rknaebel/discopy-data',
      author='Rene Knaebel',
      author_email='rknaebel@uni-potsdam.de',
      license='MIT',
      packages=find_packages(),
      install_requires=[
            'numpy>=1.18.0',
            'nltk>=3.4',
            'joblib',
            'tensorflow>=2.1.0',
            'transformers==4.2.1',
      ],
      zip_safe=False,
      )
