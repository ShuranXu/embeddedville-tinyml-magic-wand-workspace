# Third-party notices

## Arduino_BMI270_BMM150 1.2.4

The Rev2 starter vendors the unmodified `src/` directory from Arduino's
`Arduino_BMI270_BMM150` tag `1.2.4` because this tagged release is not present
in the Arduino Library Manager index used by the pinned CLI. It is distributed
under LGPL-2.1-or-later. The full license is included at
`starter/magic_wand/THIRD_PARTY_LICENSES/Arduino_BMI270_BMM150-LGPL-2.1.txt`.

## Magic Wand example

Core firmware, rasterization, model embedding, recorder concepts, and the original training workflow derive from:

- Copyright Pete Warden and contributors
- Repository: https://github.com/petewarden/magic_wand
- License: Apache License 2.0

The repository root `LICENSE` contains the Apache License 2.0 text. EmbeddedVille changes add dual-board support, reproducible profiles, a four-class project contract, data validation, current TensorFlow export, and course integration.

The course also reproduces `website/wand_attachment.jpg` from upstream commit
`cadc4aa03bd886b4f0449e943d07a88b5e871f4b` as an instructional photograph.

## TensorFlow.js TFLite WebAssembly runtime

The Virtual Wand loads pinned browser distributions of `@tensorflow/tfjs-core` 4.9.0, `@tensorflow/tfjs-backend-cpu` 4.9.0, and `@tensorflow/tfjs-tflite` 0.0.1-alpha.10 from jsDelivr. TensorFlow.js is Copyright Google LLC and contributors and is distributed under the Apache License 2.0. The runtime executes locally in the learner's browser; uploaded models and datasets are not sent to EmbeddedVille.

## TinyML book

The curriculum paraphrases concepts from *TinyML* by Pete Warden and Daniel Situnayake, Chapters 1–3 and 11–12. No book pages or figures are distributed in this repository.
