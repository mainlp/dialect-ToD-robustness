# Exploring the Robustness of Task-oriented Dialogue Systems for Colloquial German Varieties

This repository accompanies our paper *Exploring the Robustness of Task-oriented Dialogue Systems for Colloquial German Varieties* (Ekaterina Artemova, Verena Blaschke, & Barbara Plank, to be published at EACL 2024).
It contains code for automatically applying morphosyntactic perturbation rules to German sentences in order to mimic grammatical structures found in colloquial varieties (details in the paper).

## Usage conditions

We release this code for research purposes only, and expressly forbid usage for mockery or parody of any dialects or registers.


## Dialect perturbations

We implemented 18 perturbations covering a wide range of dialect phenomena in German. The code is available in the ```dialect_perturbations.py``` file and the example usage is demontrated in the ```perturbation_test.ipynb``` notebook. 

To test the perturbation, you'll require dictionaries and word lists from the ```resources``` folder, and the following packages:

- [SoMaJo](https://pypi.org/project/somajo/) for tokenization
- [SpaCy](https://spacy.io/) for POS tagging
- [Stanza](https://stanfordnlp.github.io/stanza/) for POS tagging and dependency parsing
- [DERBI](https://github.com/EducationalTestingService/DERBI) for inflection
- [Pattern-de](https://www.clips.uantwerpen.be/pages/pattern-de) for verb conjugation




# Cite us 

TBA
