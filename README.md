---
layout: page
title: Denoising with Generative Models
published: true
description: Project at VITA lab - Preliminary notes 
---

# Week 2 : Learning pytorch, finding data

## Beethoven dataset

Downloaded, ~350MB, OGG format, bitrate between 96 and 112 kbps

## MAESTRO Dataset

Piano recording from virtuosic piano performances, ~200 hours in total, available in midi (precisly recorded from the piano, 85MB) or in WAV (122 GB).

Might be a very good complement to the Beethoven dataset, and maybe better quality ? 

[here](https://magenta.tensorflow.org/datasets/maestro)
## Lakh MIDI dataset

176k+ midi files, [here](https://colinraffel.com/projects/lmd/)
## Pytorch 

Big tutorial [here](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html). 

U-net in paytorch [here](https://github.com/milesial/Pytorch-UNet). Seems really easy to creates sub-modules in a seperate file, and then call them from the main entwork. So it will be quite easy to create a class for the downsampling block and the class for the upsampling block, and then put them one after the other. Similarly, for the discriminator, they repeat a block 7 times, so we can create it and reuse it.



Note : tu add skip connections, we just need to keep the variable representing the ouput of the downsampling block, and give it to the upsampling block as, for instace, a class argument. We can then just "add" it.

ex : 

```
out16 = self.in_tr(x)
out32 = self.down(16, 32, out16)
out64 = self.down(32, 64, out32)
out128 = self.down(64, 128, out64)
out = self.up(128, 64, out128)
out = self.up(64, 32, out64)
out = self.up(32, 16, out32)
out = self.out_tr(out)
```



## Audio specific 

torchaudio seems to be able to do resampling, and can handle waveform audio. Can do many other transformations. Probably good to use this if we do super resolution, so we can generate our intput data. Tuto [here](https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html) Wasn't able to install it, always an error.

Could also try to use [Librosa](https://stackoverflow.com/questions/30619740/python-downsampling-wav-audio-file) that can open files downsampled directly.

## Paper

Will probably follow [this paper](https://bilat.xyz/vita/Adversarial.pdf) (MUGAN), but it is "under review" so there isn't any names. How will this work ?

Need to check how to train the external network

Input : fixed size audio sample from the data, going through low pass filter. They don't seem to give the input size, but they use 8 layers => 2^8 as the input size maybe ?

Downsampling : 4 filters, 1 of each size. Then is goes through PRelU (Parametric relu) : $f(x) = alpha * x for x < 0, f(x) = x for x >= 0$ And then it goes throught the Superpixel block (similar to a pooling block) which reduces the dimension by 2 and double the number of filters (alternate values, even goes in one output, odd goes in the other output). This seem straight forward.

Upsampling block : Once again we have the same 4 filters. I'm not sure how we are supposed to upsample if we have convolutional filters again. Then a dropout, the same PRelU, a subpixel block which this times interleaves two "samples" to make one larger. And then we stack with the input of the corresponding downsampling block.

# Week 1 : Audio denoising papers

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

[Link](http://bilat.xyz/vita/Algo_Speech.pdf)
 
## A Review of Adaptive Line Enhancers for Noise Cancellation (ALE.pdf)

Doesn't need recording of noise. Adaptive self-tuning filter that can spearate periodic and stochastic component. Detect low-level sin-waves in noise

[Link](http://bilat.xyz/vita/ALE.pdf)

## A review: Audio noise reduction and various techniques (Techniques.pdf)

Some filters : Butterworth filter, Chebyshev filter, Elliptical filter

[Link](http://bilat.xyz/vita/Techniques.pdf)

## Employing phase information for audio denoising (Phase.pdf)

TODO

[Link](http://bilat.xyz/vita/Phase.pdf)

## Audio Denoising by Time-Frequency Block Thresholding (Block_Threshold.pdf)

TODO

[Link](http://bilat.xyz/vita/Block_Threshold.pdf)

## Speech Denoising with Deep Feature Losses (Speech_DL.pdf)

Fully convolutional network, work on the raw waveform. For the loss, use the internal activation of another network trainned for domestic audio tagging, and environnement detection (classification network). It's a little bit like a GAN.

Most approaches today are done on the spectrogram domain, this one not. Prevents some artefacts due do IFT. Methods that are in the time domain often use regression loss between input and output wave. Here, the loss is the dissimilarity between hidden activations of input and ouput waves. Inspired from computer vision (-> Perceptual_Losses.pdf)

Details of The main network are given in papers, section II-A-a.Different layers in the classification/feature/loss network correspond to different time scales. The classification network is inspired by VGG architecture from CV, details in paper II-B-a. II-B-b explain how to transoorm activations / weights to a loss.

Train the feature loss network using multiple classification tasks (scene classification, audio tagging). Train the speech denoising using the [1] database. They used the clean speeches and some noise samples and created the training data by combining them together, then they are downsampled.

Experimental setup : compared with Wiener filterning pipeline, SEGAN, and the WaveNet based one used as a baseline. Used different score metrics (overall (OVL), the signal (SIG), and the background (BAK) scores)). It was better than all the baselines. Also evaluated with human testers, also better than the others.

Now this is for speech, and it might not work as well for general sound/music

[Link](http://bilat.xyz/vita/Speech_DL.pdf)

## Recurrent Neural Networks for Noise Reduction in Robust ASR (RNN.pdf)

SPLICE algorithm  is a model that can reduce noise by finding a joint distribution between clean and noisy data, ref to article in the paper's reference, but could not find it online for free.

We could simply engineer a filter, but it's hard, and not perfect .

Basic idea : We can use L1 norm as the loss function. This type of network is known as denoising autoencoder (DAE). Since input has variable length, we train on a small moving window

More advanced : Deep recurrent denoising audtoencoder, we add conection "between the windows" $\implies$ Input is [0 1 2] [1 2 3] [2 3 0], we give each one one to a NN with e.g. 3 hidden layer, with layer 2 recursively connected, and it gives [1] [2] [3] as the output. Uses Aurora2 corpus, with noisy variants synthetically generated

[Link](http://bilat.xyz/vita/RNN.pdf)

## Investigating RNN-based speech enhancement methods for noise-robust Text-to-Speech (RNN_Speech_Enhancement.pdf)

Shows the two alternative approaches (time vs frequency) on a graph

[Link](http://bilat.xyz/vita/RNN_Speech_Enhancement.pdf)


## Audio Denoising with Deep Network Priors (DN_Priors.pdf)

Combines time and frequency domain, unsuppervised, you try to fit the noisy audio and since we only partialy fit the output of the network helps to find the clean audio. Link to github repo with some data and the code [github](https://github.com/mosheman5/DNP).

Usually we first create a mask that tells us what frequency are noise, then we use an algo that removes those frequencies.

Here the assumption is that it is hard by definition to fit noise, so the NN will only fit the clean part of the input. 

Technique already used in CV. Diff : in CV, the output is already the cleaned image, not here, so they create a spectral mask from the ouput to then denoise the audiio. Better than other unsupervised methods, almost as good the the supervised ones.

=> Probably not usefull for GANs.

[Link](http://bilat.xyz/vita/DN_Priors.pdf)

## Spectral and Cepstral Audio Noise Reduction Techniques in Speech Emotion Recognition (Spectral_Cepstral.pdf)

They will compare their method to methods of "Spectral substraction", where you remove the noise spectrum from the audio spectrum.

Need to look in more details at "Spectral domain", "Power Spectral"; "log-sepstral", "cepstral domain", ...

One again no NN are used here, this is mostly some signal processing, so I don't think it will be very usefull.

They also talk about "accurancy measures", e.g. "Itakura distance", "Toeplotz autocorrelation matrix", "euclidian distance between two mel-frequency cepstral vectors".

Probably more informations about signal processing techniques in the references.

[Link](http://bilat.xyz/vita/Spectral_Cepstral.pdf)

## Raw Waveform-based Speech Enhancement by Fully Convolutional Networks (RawWave_CNN.pdf)

Convolutional, waveform to waveform. Mentions like most "Wiener filtering", "spectral substraction", "non-negative matrix factorisation". Also mentions "Deep denoising autoencoder" from (RNN.pdf), also see (DDAE.pdf) that they are citing.

Explain that most models use the magnitude spectrogram (-> log-power spcetra), which will leave the phase noisy (as it used the phase from the noisy signal to reconstruct the output signal). Also mentions that it is important to use the phase information to reconstruct the signal afterwards. Apparently DL based models that use the raw form often get better results.

Fully connected is not very good since it won't keep local informaton (think high frequencies). They use A "Fully convolutional network FCN" and not a CNN, see (FCN.pdf). A FCN also mean a lot less paramters.

Convolutional is considered better since we need adjacent information to make sense of frequencies in the time domain. Fully connected layers cause problems (can't moel high and low frequencies together), so that's why we don't have one at the end in a FCN (FCN = CNN without fully-connected layers). 

For the experiment, as some of the others papers, they took clean data and corrupted it with some noise (e.g. Babble, Car, Jackhammer, pink, street, white gaussian, ...)

They also mention at the end the difference between the "shift step" for the input in the case of a DNN, but it's not very clear what they did with the FCN. They say the took 512 samples from the input wave, but does it seems really low if we use e.g. 44kHz sampling for our music.

[Link](http://bilat.xyz/vita/RawWave_CNN.pdf)

## Speech Enhancement Based on Deep Denoising Autoencoder (DDAE.pdf)

They meantion a DAE where they only trained using clean speech : Clean as in and out, then when we give a noisy signal it tries to express it on the "clean subspace/basis function", they try to model "what makes a clean speech", need to look into that. This time, they use dirty-clean pairs, so they want to know "what is the statistical difference between noisy and clean.

Once again, they create their dataset by adding some noise artificially. They mention (RNN.pdf), which uses a recurrent network, this won't be the case here.

The architecture looks like a classical DNN. They stack "neural autoencoders" together, and each AE seems to be layer - non-linear - layer. They also use regularization. For training, they first pretrain each AE individually which adequate parameters, then put them together and train again.

Measurement are specific to speech, they use "noise reduction", "speech distortion" and "perceptual evalutation for speech qualty - PESQ" / not clear what this is.

For the features they use "Mel frequency power spectrum (MFP)" on 16ms intervals

Their results are mostly better than traditional methods.

[Link](http://bilat.xyz/vita/DDAE.pdf)

## SEGAN: Speech Enhancement Generative Adversarial Network (Speech_GAN.pdf)

As some other papers, mention that most use spectral form, but here they use the raw waveform.

Explains GAN : The generator which creates some data by learning the real data distribution and trying to approximate it, and the discriminator, usually a binary classifier, that tries to tell us if our sample is a real one or one generated by the generator. The goal for the generator is to fool the discriminator.

To train : D back-props a batch of real examples classified as "true", and then a batch of fake example (generated by G) and mark them as "false". Then we fix D's parameters, and G does the backpropagation with the false example to try to make D missclassify. They then give more mathematical details and techniques (e.g. LSGAN).

They use a fully convolutional network (FCN), with a encoder-decoeer layout, were the signal is "compressed", concateneted with the latent representation (?) and then decoded. They also use skip connections so we don't lose some details about the structure (we have speech in and out some we have some similarities). e.g. they transmit phase and alignment information. They then use some information from the D network to create their loss.

Their dataset is the usual one we saw previously [1], and they use both artificial and natural noise to create their tran/test set. THey use a sliding window of the raw data (downsampled a little bit), and they also used a minor high freqency filter.

All the code is on [github](https://github.com/santi-pdp/segan). Results are positive and are mostly done by people's opinions.

[Link](http://bilat.xyz/vita/Speech_GAN.pdf)

## A Wavenet for Speech Denoising (WaveNet.pdf)

They first present the WaveNet network, which was used to synthesize natural sounding speech.

Their model is similair to WaveNet, but the convolution is "symetrically centerd" since we know both future and passed data unlinke for speech generation. They also have a different loss function, the output is not a probability but the clean data directly, 

[Link](http://bilat.xyz/vita/WaveNet.pdf)

# Audio super-resolution papers

## Audio Super-Resolution using Neural Nets (SuperRes_NN.pdf)

Paper + webpage + github on super resolution with deep networks
[https://kuleshov.github.io/audio-super-res/#](https://kuleshov.github.io/audio-super-res/#)

Supervized model on low/high quality pairs, deep convolutional network, doesn't need specialized audio processing techniques.

Explain that processing raw audio is usefull but computationally intensive.

Model is fully feed-forward and inspired from image super-resolution. They consider the sample rate as the resolution we want to improve. it will work on raw audio and (bonus) is one of the rare paper that also tried to work with non-speech audio.

Architecture : successive downsampling and upsampling blocks, each doing a convolution + batch norm + ReLU. Called "Bottleneck architecture", similar to the "autoencoders from previous papers. Also have some skip connectionn between "similar layers". This seems very similar to SEGAN.

They also use something called "Subpixel shuffling layer", it seems to be what is used in the upsampling block to half the number of filters while increasing the spatial dimension.

Two datasets, one piano [5], one with voices [6], with noisy version automaticaly generated with Chebyshev filter. They give a few metrics (Signal to Noise ratio SNR, log-spectral distance LSD)

Results are good, and the model is fast enough for real-time on their GPU, but slow to train.

When they tried with a more diverse musical dataset, it wasn't sucessfull and the model was underfitting.

[Link](http://bilat.xyz/vita/SuperRes_NN.pdf)

## Adversarial Audio Super-resolution with Unsuppervised Feature Losses (Adversarial.pdf)

Called MU-GAN
GAN are hard to train, people sometimes replace the sample-space loss with a feature loss (instead of distance between two samples in the ssample space, we use the feature maps of an auxiliary nn).

Their explanations of a GAN is very similar to the one for SEGAN. They have the generator (G), the discriminator (D), and they also have the convolutional autoencoder (A) that they use to create a loss (see first paragraph above). A is unsupervised

The generator is a convolutional u-net (as a few other papers), where each level works has many filter sizes, and we have skip-connections. They use "Subpixel nlocks" in the upsampling block, and the opposite, a "superpixel block" in the downsampling block. Subpixels were mentionend in the previous paper, and superpixel lets you decrease the dimensionality, a little like pooling/strided convolutions. They have good illustrations explaining this.

For the loss, they use the simple L2 loss from the G network, the adversarial loss from the D network. This alone doesn't give better results that a model with no GAN, so they also use the feature loss generated by the A network.

They used the [6] dataset for voices, and [5] for the piano dataset. As metrics, signal-to-noise-ratio (SNR), log-spectral distance (LSD) and mean opinion score (MOS). Good results. A being unsuppervised is on-par to the classifier based loss.

[Link](http://bilat.xyz/vita/Adversarial.pdf)

## Time Series Super Resolution with Temporal Adaptive Batch Normalization (TimeSerie_Batch.pdf)

[Link](http://bilat.xyz/vita/TimeSerie_Batch.pdf)

They want to combine recurent and convolutional, in a new type of layer called "temporal adaptive normalization". This allows the filters to be turned on and off by long range information coming from the recurrent part. Some illustration explain the architecture in more details, but once again it looks like a u-net.

Super-resolutation has some spatial invariance, which implies the CNN, but we still might some usefull information furhter away, so the recurrent part can help us with that.

Need to look in more details at this layer. They also use the [5] and [6] datasets

# Ideas from images

## Perceptual Losses for Real-Time Style Transfer and Super-Resolution (Perceptual_Losses.pdf)

[Link](http://bilat.xyz/vita/Perceptual_Losses.pdf)

## The Unreasonable Effectiveness of Deep Features as a Perceptual Metric (Perceptual_Metric.pdf)

[Link](http://bilat.xyz/vita/Perceptual_Metric.pdf)

## Fully Convolutional Networks for Semantic Segmentation (FCN.pdf)

[Link](http://bilat.xyz/vita/FCN.pdf)

# Datasets 

## 1: Voice database with noisy and clean version

[https://datashare.is.ed.ac.uk/handle/10283/1942](https://datashare.is.ed.ac.uk/handle/10283/1942)

## 2: New version of [1], also voice

[https://datashare.is.ed.ac.uk/handle/10283/2791](https://datashare.is.ed.ac.uk/handle/10283/2791)

## 3: Speech database with clean and noisy

[https://github.com/dingzeyuli/SpEAR-speech-database](https://github.com/dingzeyuli/SpEAR-speech-database)

## 4: Aurora2

[http://aurora.hsnr.de/aurora-2.html](http://aurora.hsnr.de/aurora-2.html)

Some script that can generate noisy data

## 5: Piano dataset Beethoven

Not sure where to find it as the link was dead, but it might be this [https://gist.github.com/moodoki/654877be611ef63bb32d58c428d6e7ba](https://gist.github.com/moodoki/654877be611ef63bb32d58c428d6e7ba)

## 6: CSTR VCTK Corpus

[https://datashare.is.ed.ac.uk/handle/10283/2651](https://datashare.is.ed.ac.uk/handle/10283/2651)

## 7: Magnatagatune dataset

[http://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset](http://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset) 200 hours of music from 188 different gernres.
