# Agent Notes

- Do not hallucinate repository behavior.
- Always inspect files before making changes.
- Keep external repositories as close to original as possible.
- Put PLATO-specific scripts in `plato/scripts/`.
- Put PLATO-specific configs in `plato/configs/`.
- Put experiment outputs in `results/`.
- Do not put large datasets into git.
- Local dataset root is `/home/amao/datasets`; current datasets are EuRoC and LIASuw. Read `plato/docs/dataset_guide.md` before writing dataset-specific scripts.
- Do not run long builds unless explicitly requested.
- Use `rg` for text search when available.
- Prefer lightweight checks such as `ls`, `find`, `tree`, and `git status`.
- If an external repository must be modified, document the reason and exact changed files.

After each meaningful change, update:

- `AGENTS.md`
- `plato/docs/project_memory.md`
- `plato/docs/repository_map.md` if repository structure changes
- `plato/docs/integration_plan.md` if pipeline changes
