# Welcome to Week 1
Aims:
- Reviews FNN/ CNN <-> modular code.
- Learning to use *packaging* (multi-file, import, CLI), for a big project

Smaller Tasks (If you need):
- Clone a template repo (Poetry/pipenv, pre-commit).
  - However, I prefer using conda due to its convenient, and easily to manage global packages. Then I just need to define environment.yml.
- Review FFN, CNN by building a simple project classify MNIST data.
  - Project folder include:
    - `models/`, `data/`, `train.py`.
- Add `__main__.py` to run `python -m yourpkg train ...`.
- Unit-test a simple loss (pytest)

Remember, these are just the template tasks, you can modify things that best suit you.

## Init Tasks
- Now start by create an environment, install needed packages then export to an environment.yml file in the root folder

- Now learn to build a package system using __init__.py

repo_root/
├─ src/
│   └─ weekly_projects/
│       ├─ __init__.py
│       ├─ neural_network/
│       │    ├─ ffn.py
│       │    └─ cnn.py
│       ├─ encoder/
│       │    └─ bert_encoder.py
│       └─ decoder/
│            └─ gpt_decoder.py
└─ pyproject.toml

This is the order tree for you to understand how we code packages for big projects

We can call out our package by using `from weekly_projects.models.ffn import FFN` kinda like this

- Okay, we just learned new things about packages in "clean code" setup for a big project, that a big step. Now we need to jump into building our projects

## Week 1 Tasks
### `models/ffn.py`
- Build a FFN at least 2 Linear -> ReLU -> Linear
- Question: What Interface to swap between CNN and FFN without modifying trainer?

### `data/mnist.py`
- Get into torch, using `get_loaders(batch=64)` ,`train_.loader` and `val_loader`
- Question: Where do you use `torchvision.datasets.MNIST` to install and get batches ?

### `trainer.py`
- Function `train_one_epoch(model, loader, optim, loss_fn)`
- Question: Should you out avg_loss or just print it?

### `cli.py`
- Run: `python -m weekly_project train --epochs 3 --batch 64`
- Question: *Sub-command* `train` vs flag `--mode` -- What do you prefer?

### `tests/test_loss.py`
- Pytest to confirm cross-entropy <= 1e-4 for toy?
- You use algorithm or using `torch.nn.functional.cross_entropy` for oracle?
