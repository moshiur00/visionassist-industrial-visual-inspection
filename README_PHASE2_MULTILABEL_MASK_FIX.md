# Phase 2 multi-label mask fix

Extract this archive directly into the project root and overwrite matching files.

VisA PCB masks may contain indexed foreground values such as 1, 2, and 3. These are valid semantic/instance labels, not invalid grayscale noise. The parser now:

- treats every pixel value greater than zero as anomaly foreground;
- preserves the original unique mask values;
- records whether the source mask is already binary;
- accepts multi-label indexed masks;
- reports source-binary, multi-label, and binary-compatible mask counts separately.

Run:

```powershell
uv run pytest tests/test_parse_visa.py
uv run visionassist phase2-visa --config configs/data/visa.yaml
```
