PROBLEM STATEMENT 12

Fill in the Frames Seamlessly - Enhancing Temporal Resolution of Satellite Imagery using the techniques of AI/ML based on Optical Flow
Description
Satellite images for geo-stationary satellites are often captured at fixed intervals (e.g., every 30 minutes for geostationary satellites like INSAT, every 10 minutes for Himawari/GOES satellites or every few days for polar-orbiting satellites). This leads to limitations in temporal resolution that restricts near real-time monitoring of dynamic phenomena such as fire, cyclones, thunderstorms, floods and rapid land changes. Traditional optical-flow based temporal interpolation is limited with incorrect results (blurred images & other artefacts) and they fail to capture fast-moving or non-linear changes in cloud dynamics.

This problem proposes to develop an AI/ML based Optical Flow frame interpolation technique to generate intermediate frames between two consecutive satellite images to effectively enhance temporal resolution, enabling more frequent observations without additional satellite resources.

Objective
The developed software should include and accomplish the following steps:

Develop an optical flow-based model to estimate motion vectors between consecutive satellite frames.
Generate synthetic intermediate frames using Deep Learning frame interpolation methods.
Improve temporal resolution of satellite imagery (e.g., from 30 minutes to 15 minutes and further to 7.5 minutes interval or equivalent).
Validate results against real higher temporal resolution datasets (like Himawari/GOES-19) using image similarity metrics such as SSIM, MSE, PSNR, FSIM, etc.
Expected Outcomes
Generation of synthetic intermediate frames using Deep Learning based frame interpolation methods with Optical Flow for INSAT-3DS/3DR imagery.

Dataset Required
Thermal Infrared band data (around 10 micrometer) of geo-stationary satellite datasets to be used from online sources. The input data should be .nc or .h5 files.

GOES-19 ABI Channel 13 data from NOAA GOES-19 AWS bucket
INSAT-3DS/3DR TIR1 channel data from mosdac
Himawari-8.
Suggested Tools/Technologies
Deep Learning based video interpolation models can be explored and trained/fine-tuned using the above datasets (e.g., Super SloMo, RIFE, etc).
Web technologies to be used for visualisation (Step 2 of Expected Solution).
Expected Solution / Steps to be followed to achieve the objectives
Solution for the challenge should include a prototype for AI-powered frame interpolation for satellite imagery that should comprise of:

Frame Interpolation
Takes consecutive satellite images and generates intermediate frames. Should include:
Optical Flow Estimation: can explore and use AI/ML models. Preferably, models are to be trained on satellite data collected from Step A.
Frame Interpolation: can explore and use interpolation networks for synthesis of new frames.
The input and output files to the model should be .nc files.
Visualisation
Develop a dashboard showing original vs. interpolated satellite animations. Should include:
After interpolation of a set of satellite frames, the output should be displayed with time-lapse animations.
Animations to be present both for ground truth (original satellite frames) and also for interpolated frames.
Generate a report comparing results with ground truth (existing frames). Should include:
Comparison of generated frames with the ground truth using metrics of SSIM, MSE, FSIM, etc. Include plots of the results
Any other suitable metrics can also be explored and used. Metrics should be suitable to capture the cloud movements.
INSAT-3DS
Model to be applied on INSAT-3DR/3DS data.
Based upon the best model created/used/trained from above steps, the intermediate frames of the INSAT-3DS are to be interpolated and the animations are to be prepared for the INSAT data at a temporal resolution of 15 minutes.
For Steps 1-3, GOES-19 data or any other geo-stationary satellite of high temporal frequency, may be used, e.g., taking the frames of 00.00 hrs and 00.20 hrs as input, the model should output the frame at 00.10 hrs.

For Step 4, the above trained model(s) have to be used on INSAT-3DS data.

Evaluation Parameters
The evaluation would be done on:

Frame Interpolation: Image quality metrics of MSE, PSNR and any other image metrics of how the generated frame is different from the ground truth
Visualisation: Web GUI Design
Subject to Visual Quality of the INSAT-3DS interpolation images.
