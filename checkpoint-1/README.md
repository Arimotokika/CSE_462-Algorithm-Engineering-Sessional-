# Checkpoint 1 README

This folder contains the Checkpoint 1 deliverables for the WTA sessional.

Checkpoint 1 focuses on:

- formal problem definition,
- hardness and reduction-based discussion,
- survey of existing algorithm families,
- early demonstration content before the full Checkpoint 2 implementation.

## Files

- `guidelines.txt` - official checkpoint requirements.
- `presentation.tex` - Checkpoint 1 slide deck source.
- `images/` - supporting visual assets used by the slides.

## Intended Coverage

Checkpoint 1 content is organized to satisfy the early milestone requirements:

- define the WTA problem and motivation,
- discuss hardness/complexity,
- present survey-level algorithm context,
- prepare groundwork for Checkpoint 2 implementation.

## Build Instructions

Compile inside this folder:

```bash
pdflatex presentation.tex
pdflatex presentation.tex
```

Two passes are recommended for stable table-of-contents and references.

Generated file (after compile):

- `presentation.pdf`

## Scope Note

This is an archived milestone. The final implementation, experiments, and reproducibility pipeline are documented in:

- `../checkpoint-2/README.md`
- `../codes/README.md`
