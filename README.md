# Trust-aware RL in the Semantic Memory Game

This repository contains code implementing trust-aware RL as described in
"Human-AI Collaboration via Trust Factors: a Collaborative Game Use Case".

## Installation

The implementation requires python 3.10+, and the libraries listed in
`requirements.txt`. These can be installed via `pip` by issuing

`pip3 install -r requirements.txt`

## Running

The python script in `semantic_memory/mixed_team.py` is the main entry point
for the implementations. It can be executed from the command line, while being
inside the `semantic_memory` directory, as

`python3 mixed_team.py <command>`

where `<command>` may be one of `train`, `eval`, `print`, `plot`, or `test`,
depending on the desired task.

## Acknowledgments

The development of this software is supported by the Air Force Office of
Scientific Research under award number FA8655-23-1-725 and PNRR MUR project
PE0000013-FAIR.

## Citation

```
@misc{fanti-2025-humanai,
    author = {Fanti, Andrea and Frattolillo, Francesco and Laudati, Rosapia and Patrizi, Fabrizio and Iocchi, Luca},
    title = {Trust-aware RL in the Semantic Memory Game},
    year = {2025,
    publisher = {GitHub},
    journal = {GitHub repository}}
}
```


