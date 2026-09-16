# Contributing

For bug reports, include the command, dependency versions, checkpoint manifest,
image resolution, rotation sampler and whether PNG round-tripping was enabled.
Do not attach private panoramas; a shareable synthetic example is sufficient.

Run `python -m unittest discover -s tests -v` before submitting changes.
Changes to learned modules must retain strict checkpoint compatibility or
explicitly version the model configuration. Changes to sampling or distortion
parameters must document how evaluation results change.
