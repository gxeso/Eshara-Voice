\# Eshara Voice



Eshara Voice is a real-time Emirati Sign Language-to-speech prototype built with Python, MediaPipe, OpenCV and machine learning.



The project currently recognizes a small healthcare and emergency vocabulary using webcam-based hand landmark tracking. Recognized signs are displayed on screen and spoken using the Windows speech synthesizer.



\## Current vocabulary



\- Help

\- Doctor

\- Medicine

\- Pharmacy

\- Emergency

\- Pain

\- Hospital Ward



\## Technologies



\- Python

\- OpenCV

\- MediaPipe

\- NumPy

\- scikit-learn

\- Random Forest

\- Windows System.Speech



\## How it works



1\. OpenCV captures webcam frames.

2\. MediaPipe detects 21 landmarks on each hand.

3\. Each two-second sign is converted into a 30-frame sequence.

4\. Missing detection frames are interpolated.

5\. A trained Random Forest predicts the sign.

6\. The result is displayed and spoken aloud.



\## Installation



```bash

python -m venv .venv

.venv\\Scripts\\activate

python -m pip install -r requirements.txt

