# STREP - Sustainable and Trustworthy Reporting for ML

Software repository for more sustainable and trustworthy reporting of ML results - associated research paper currently under review.

## Just access our public webapp - coming soon

## Installation
Simply install required packages via `pip install -r requirements.txt` (tested with Python 3.8).

## Usage
Run `main.py` to process and ivnestigate the pre-assembled databases (startup might take some time).
If you want to explore your own database, simply pass the path to the pickled `pandas` dataframe via `main.py --database [path/to/your/]database.pkl`.
If you want to also process meta information with STREP, just include `json` files in the database directory - make sure to follow our convention as given in the pre-assembled database directories (documentation to follow).

## Repository Structure
The `databases` directory contains different gathered databases of ML reports, including scripts to assemble some of them.
`strep` contains code that processes these databases, calculates index values and compound scores for measurements, and visualizes them.
All `paper_results` can be found in the respective folder, and can be re-generated with the corresponding script.
`parse_logs.py` allows to generate a database file from multiple log directories (documentation to follow).

## Terms of Use
Copyright (c) 2023 Raphael Fischer