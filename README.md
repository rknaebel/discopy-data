# Datasets and Preprocessing for Discourse Data

## Setup
You can easily install *discopy-data* by using pip:
```shell script
pip install git+https://github.com/rknaebel/discopy-data
```
or you just clone the repository.
The you can either install discopy-data through pip
```shell script
pip install -e path/to/discopy-data
```

## Usage
*Discopy-data* is the discopy backend that handles datastructures, preprocessing, and dataset extraction.

### Sample preparation of a text file, adds also constituent parse trees
The first script uses trankit for tokenization, tagging, and dependency parsing.
In addition, the second script is used, to add constituency trees with the supar parser.
If dependency trees should be added by super as well, add the flag `-d`.
```shell script
discopy-tokenize -i /some/examples/wsj_0336 | discopy-add-parses -c
```
### Tokenize raw text without tagging nor parsing
This might be useful for neural pipeline that does not rely on language features.
```shell script
cat /some/text | discopy-tokenize --tokenize-only
```
### Preparation of full datasets
This is still experimental. A list of possible datasets is listed under `cli/extract.py`.
```shell script
discopy-extract pdtb /data/discourse/conll2016/ --use-gpu --limit 2 | discopy-add-annotations pdtb /data/discourse/conll2016/ --simple-connectives --sense-level 2 | discopy-update-parses --dependency-parser ''
```