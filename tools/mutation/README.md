# Manual Mutation Tools

These scripts are retained as explicit development tools only. The runtime
topology and `quantale_semiring_v2` binary do not dispatch them.

Use them manually from the repository root after setting:

```bash
QUANTALE_ENABLE_MUTATION_TOOLS=1
```

Example:

```bash
python3 tools/mutation/apply_mutations.py --list
```
