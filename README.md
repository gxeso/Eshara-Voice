# Eshara Voice

Eshara Voice is a camera-based Emirati Sign Language-to-speech prototype
focused on healthcare and emergency communication.

The application detects hand movements through an ordinary webcam, recognizes
supported signs, displays their meaning and speaks the result aloud.

## Current status

Version: **Prototype v0.1**

The current model supports 10 healthcare and emergency signs studied using
the official UAE Sign Language Dictionary.

Current vocabulary includes:

- Help
- Doctor
- Medicine
- Pharmacy
- Emergency
- Pain
- Hospital Ward
- Additional healthcare signs included in the latest model

## Features

- Real-time webcam input
- Two-hand landmark detection
- 21 landmarks per hand
- Temporal sign capture
- Missing-frame interpolation
- Machine-learning classification
- Confidence scoring
- Readable multiword labels
- Windows text-to-speech output
- Offline recognition after setup

## Technology

- Python
- OpenCV
- MediaPipe
- NumPy
- scikit-learn
- Random Forest
- Windows System.Speech
- Object-oriented architecture

## Processing pipeline

1. OpenCV captures live webcam frames.
2. MediaPipe identifies hand landmarks.
3. Each sign is captured over two seconds.
4. Frames are resampled into a `30 × 132` sequence.
5. Temporary landmark-detection failures are interpolated.
6. The trained model classifies the sequence.
7. The recognized result is displayed and spoken.

## Dataset

The current prototype was trained using original landmark recordings created
specifically for this project.

The private dataset, rejected samples and backup recordings are not included
in the public repository.

Initial held-out testing produced strong results, but those results come from
a controlled, primarily single-signer dataset. They should not be interpreted
as proof of real-world or universal accuracy.

Planned evaluation includes:

- Multiple signers
- Unseen participants
- Different lighting conditions
- Different clothing and backgrounds
- Precision, recall and confusion-matrix analysis
- Deaf-community and interpreter feedback

## Sign-language reference

The supported signs were studied using the official UAE Sign Language
Dictionary published by the Zayed Authority for People of Determination.

This repository does not redistribute the dictionary's videos, images, audio
or other media. Eshara Voice is independent and is not affiliated with or
endorsed by the Zayed Authority or the UAE Government.

## Important limitation

Eshara Voice currently recognizes a limited vocabulary. It is not a complete
Emirati Sign Language translator, certified accessibility product or
replacement for qualified interpreters.

## Roadmap

- Unknown/no-sign recognition
- Expanded healthcare vocabulary
- Arabic and English speech
- Multi-user dataset
- Sequence-based neural model
- Continuous recognition
- Desktop and mobile interfaces
- Community-led validation
- Privacy-focused on-device deployment

## Author

**Ghassan Madi**  
University of Wollongong in Dubai

## Copyright

Copyright © 2026 Ghassan Madi. All rights reserved.

No permission is granted to copy, modify, redistribute or commercially use
this project without prior written permission.