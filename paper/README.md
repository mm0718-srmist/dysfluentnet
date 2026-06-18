# Paper source

`dysfluentnet_camera_ready.tex` is the Interspeech 2026 camera-ready source
for *DysfluentNet: Joint Stuttering Event Detection and Dysfluency-Aware
Transcription via Hierarchical Self-Supervised Learning*.

## Still needed to compile

This file alone will not compile yet -- three things referenced by the
`.tex` source were not part of this upload and need to be added here
before running `pdflatex`/`latexmk`:

- `mybib.bib` -- the BibTeX database (`\bibliography{mybib}`, IEEEtran
  style). Add your existing verified `.bib` file with the ~25 entries used
  in the paper (Lea et al. SEP-28k, Chen et al. WavLM, Ratner & MacWhinney
  FluencyBank, etc.).
- `fig_architecture_3.pdf` -- the model architecture figure
  (referenced in the Methods section).
- `fig_layerweights.pdf` -- the layer-attention weights figure
  (referenced in the Results/Analysis section).
- The official Interspeech 2026 style files (`Interspeech.cls`,
  `Interspeech.sty`, etc.) if not already on your local TeX path --
  available from the conference's author kit.

## Compiling

```bash
cd paper
pdflatex dysfluentnet_camera_ready.tex
bibtex dysfluentnet_camera_ready
pdflatex dysfluentnet_camera_ready.tex
pdflatex dysfluentnet_camera_ready.tex
```
