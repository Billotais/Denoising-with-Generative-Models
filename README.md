---
layout: page
title: Denoising with Generative Models
published: true
description: Project at VITA lab - Preliminary notes 
---

# Final README

## Introduction

**Description of the problem**

The quality of audio recordings from a mobile device has gotten better over the years, but there are still a lot of factors that can decrease the quality :

- Size and quality of the microphone sensor
- Location of the microphone compared to the audio source
- Shape of the room (causing some reverberation)
- Obstruction of the microphone (phone case, hand, ...)
- Ambient noise (voices, traffic, rain, …)

It would be useful to overcome those limitations by using software tools that would be able to automatically improve the audio quality of a given audio sample.

**Why is it important ?**

If we want high-quality recording on our mobile devices, we need some software solutions. as we might not be able to improve the hardware quality of the microphone due to physical limitations. It is also hard to control the environment where we want to do our recording. This type of technology could then be used by smartphone manufacturers to let the users create studio-grade quality recordings.

Moreover, If we are able to improve the quality of an audio signal, we might also be able to improve the quality of other types of signal (e.g. an electromagnetic signal). 

For instance, it could be used to improve the precision of the LIDAR technology that can be very useful for autonomous cars.

**Precise problem statement**

Given a music sample of any length and recorded using low quality equipment in a noisy environment (therefore it might have a low resolution, some noise and some reverberation) : 

- Output a higher resolution version of the same audio sample, with some of the noise and reverberation removed.
I- f the resulting music file sounds better to the human ear than the original, the transformation is considered successful. 



## Architecture

### Original Architecture

### GAN

### Collaborative GAN

## Code

### How to run

### How to edit

## Experiments

### Datasets

### Results

## Potential improvements

