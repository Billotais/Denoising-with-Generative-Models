# Audio denoising papers

## Noise Reduction Techniques and Algorithms For Speech Signal Processing (Algo_Speech.pdf)

Different types of noise : 

- Background noise
- Echo
- Acoustic / audio feedback (Mic capture loudspeaker sound and send it back)
- Amplifier noise
- Quantization noise when transformning analog to digital (round values), neglectable at sampling higher than 8kHz/16bit 
- Loss of quality due to compression

Linear filterning (Time domain) : Simple convolutation 

Spectral filtering (Frequency domain) : DFT and back

ANC needs a recording of the noise to compare it to the audio

Adaptive Line Enhancer (ALE) doesn't need it.

Smoothing : noise is often random and fast change, so smoothing can help again white and blue (high freq) noise.
 
## A Review of Adaptive Line Enhancers for Noise Cancellation (ALE.pdf)

Doesn't need recording of noise. Adaptive self-tuning filter that can spearate periodic and stochastic component. Detect low-level sin-waves in noise

...

## A review: Audio noise reduction and various techniques (Techniques.pdf)

Some filters : Butterworth filter, Chebyshev filter, Elliptical filter

## Employing phase information for audio denoising (Phase.pdf)



## Audio Denoising by Time-Frequency Block Thresholding (Block_Threshold.pdf)

## Speech Denoising with Deep Feature Losses (Speech_DL.pdf)

Fully convolutional network, work on the raw waveform. For the loss, use the internal activation of another network trainned for domestic audio tagging, and environnement detection (classification network). It's a little bit like a GAN.

Most approaches today are done on the spectrogram domain, this one not. Prevents some artefacts due do IFT. Methods that are in the time domain often use regression loss between input and output wave. Here, the loss is the dissimilarity between hidden activations of input and ouput waves. Inspired from computer vision (-> Perceptual_Losses.pdf)

Details of The main network are given in papers, section II-A-a.Different layers in the classification/feature/loss network correspond to different time scales. The classification network is inspired by VGG architecture from CV, details in paper II-B-a. II-B-b explain how to transoorm activations / weights to a loss.

Train the feature loss network using multiple classification tasks (scene classification, audio tagging). Train the speech denoising using the [1] database. They used the clean speeches and some noise samples and created the training data by combining them together, then they are downsampled.

Experimental setup : compared with Wiener filterning pipeline, SEGAN, and the WaveNet based one used as a baseline. Used different score metrics (overall (OVL), the signal (SIG), and the background (BAK) scores)). It was better than all the baselines. Also evaluated with human testers, also better than the others.

Now this is for speech, and it might not work as well for general sound/music

## Recurrent Neural Networks for Noise Reduction in Robust ASR (RNN.pdf)

## Investigating RNN-based speech enhancement methods for noise-robust Text-to-Speech (RNN_Speech_Enhancement.pdf)
## Audio Denoising with Deep Network Priors (DN_Priors.pdf)

## Spectral and Cepstral Audio Noise Reduction Techniques in Speech Emotion Recognition (Spectral_Cepstral.pdf)

## Raw Waveform-based Speech Enhancement by Fully Convolutional Networks (RawWave_CNN.pdf)

## SEGAN: Speech Enhancement Generative Adversarial Network (Speech_GAN.pdf)

## A Wavenet for Speech Denoising (WaveNet.pdf)

# Audio super-resolution papers

## Audio Super-Resolution using Neural Nets (SuperRes_NN.pdf)

Paper + webpage + github on super resolution with deep networks
https://kuleshov.github.io/audio-super-res/#

## Adversarial Audio Super-resolution with Unsuppervised Feature Losses (Adversarial.pdf)

## Time Series Super Resolution with Temporal Adaptive Batch Normalization (TimeSerie_Batch.pdf)

# Ideas from images

## Perceptual Losses for Real-Time Style Transfer and Super-Resolution (Perceptual_Losses.pdf)

## The Unreasonable Effectiveness of Deep Features as a Perceptual Metric (Perceptual_Metric.pdf)

# Datasets 

## 1: Voice database with noisy and clean version

https://datashare.is.ed.ac.uk/handle/10283/1942 

## 2: New version of [1], also voice

https://datashare.is.ed.ac.uk/handle/10283/2791 

## 3: Speech database with clean and noisy


https://github.com/dingzeyuli/SpEAR-speech-database
