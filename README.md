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

## Recurrent Neural Networks for Noise Reduction in Robust ASR (RNN.pdf)

## Audio Denoising with Deep Network Priors (DN_Priors.pdf)

## Spectral and Cepstral Audio Noise Reduction Techniques in Speech Emotion Recognition (Spectral_Cepstral.pdf)



# Audio super-resolution papers

## Audio Super-Resolution using Neural Nets (SuperRes_NN.pdf)

Paper + webpage + github on super resolution with deep networks
https://kuleshov.github.io/audio-super-res/#

## Adversarial Audio Super-resolution with Unsuppervised Feature Losses (Adversarial.pdf)

## Time Series Super Resolution with Temporal Adaptive Batch Normalization (TimeSerie_Batch.pdf)
# Datasets 

## Speech database with clean and noisy

https://github.com/dingzeyuli/SpEAR-speech-database
