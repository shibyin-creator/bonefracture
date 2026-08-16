# Dataset

After `python scripts/build_dataset.py --images 20000` this folder holds **20,000** YOLO-labeled images.

```
dataset/
  data.yaml
  metadata.csv
  summary.json
  images/train|val|test/*.jpg
  labels/train|val|test/*.txt
```

## Counts (default)

- Total: 20,000
- Train: 16,000 (80%)
- Val: 2,000 (10%)
- Test: 2,000 (10%)
- Detection classes: 7 (abstract)
- Morphology labels: 10 (base paper) in `metadata.csv`

## Real clinical ≥20k merge

See `docs/PROPOSAL.md`. Put YOLO exports in `dataset/external/` and run `python scripts/merge_external.py`.
