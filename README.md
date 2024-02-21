# Exploring the Robustness of Task-oriented Dialogue Systems for Colloquial German Varieties

This repository accompanies our paper [*Exploring the Robustness of Task-oriented Dialogue Systems for Colloquial German Varieties*](https://arxiv.org/abs/2402.02078) (Ekaterina Artemova, Verena Blaschke, & Barbara Plank, to be published at EACL 2024).
It contains code for automatically applying morphosyntactic perturbation rules to German sentences in order to mimic grammatical structures found in colloquial varieties (details in the paper).

## Usage conditions

We release this code for research purposes only, and expressly forbid usage for mockery or parody of any dialects or registers.


## Dialect perturbations

We implemented 18 perturbations covering a wide range of dialect phenomena in German. The code is available in the ```dialect_perturbations.py``` file and the example usage is demontrated in the ```perturbation_test.ipynb``` notebook. 

To test the perturbation, you'll require dictionaries and word lists from the ```resources``` folder, and the following packages:

- [SoMaJo](https://pypi.org/project/somajo/) for tokenization
- [SpaCy](https://spacy.io/) for POS tagging
- [Stanza](https://stanfordnlp.github.io/stanza/) for POS tagging and dependency parsing
- [DERBI](https://github.com/maxschmaltz/DERBI) for inflection -- at the moment, the [2022 version](https://github.com/maxschmaltz/DERBI/tree/e95634eba3aee5d9d2e15440f489ba98b7a9d04c) is needed for the code to run (integerated as a submodule here)
- [Pattern-de](https://github.com/clips/pattern/) for verb conjugation

### Installation
```
pip install somajo
pip install stanza
pip install spacy
python -m spacy download de_core_news_sm
pip install pattern
```

## Human evaluation

The table in the ```human_eval``` folder contains results of the **human evaluation of perturbations on the Likert scale from 1 to 5**. Each row corresponds to a pair of sentences where one sentence is a perturbation of the other. The columns are as follows:

- `sentence`: the intact sentence 
- `perturbed_sentence`: the perturbed sentence 
- `perturbation`: the perturbation applied 
- `ann_x`: the score from the annotator `x`
- `ann_y`: the score from the annotator `y`.


## Results 

The folder `plots` contains plots used in the main part of the paper and Appendices C and D.

The folder `results` contains resulting tables. Each table contains intent accuracy and slot F1 values for intact and perturbed test sets. 

We use the following convention to name files. Each file is named according to the pattern '{train language}{dev language}.{test language}.{dataset}'. The suffix '1p' denotes cases where single perturbations are applied. In other cases, all perturbations are applied simultaneously by default.

## Cite us 


```
@inproceedings{artemova-etal-2024-exploring,
  author    = {Artemova, Ekaterina and Blaschke, Verena and Plank, Barbara},
  title     = {Exploring the Robustness of Task-oriented Dialogue Systems for Colloquial German Varieties},
  booktitle = {Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics},
  year      = {2024},
  publisher = {Association for Computational Linguistics},
  note      = {To appear},
}
```
