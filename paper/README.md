# PLATO Paper

The PLATO paper is managed as a git submodule at:

```text
paper/plato-overleaf
```

The submodule URL is the Overleaf git remote:

```text
https://git@git.overleaf.com/69a927ff71bb6f987df27546
```

Use this folder for manuscript source and figures. Keep PLATO code, scripts, configs, datasets, and experiment outputs outside the paper submodule.

Useful commands from the PLATO root:

```bash
git submodule update --init --recursive
git -C paper/plato-overleaf status
git -C paper/plato-overleaf pull
```

Note: when the submodule was added, the original local Overleaf checkout at `/home/amao/Writing Paper/PLATO/69a927ff71bb6f987df27546` had uncommitted edits and LaTeX build artifacts. Those uncommitted changes are not automatically part of the submodule clone.
