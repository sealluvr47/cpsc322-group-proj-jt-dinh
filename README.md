# CPSC 322 Group Project — JT Dinh

## Project overview

This project proposes a binary classifier for brain aneurysm detection in people with autosomal dominant polycystic kidney disease (ADPKD). ADPKD is an inherited kidney condition associated with brain aneurysms. Research has also examined aneurysms in people who received kidney transplants for ADPKD, motivating an investigation of whether transplant history adds predictive value.

## Research question

Among people with ADPKD who underwent brain imaging, can we classify whether an aneurysm was detected using information available before the scan?

## Prediction target and dataset

Each row would represent one screened person with ADPKD. The target is **“Was a brain aneurysm detected on screening?”**, labeled **yes** or **no** based on the scan result.

Potential inputs, if recorded before screening, include:

- Age
- Blood pressure
- Kidney function
- Prior kidney transplant history

To evaluate whether transplant history improves predictions, the dataset must include people both with and without a prior transplant. People who were never screened have an unknown outcome and should not be labeled as negative cases.

## Project scope

The proposed task predicts aneurysm detection at screening, rather than future aneurysm development or rupture. Dataset availability and the proposed input variables still need to be confirmed.
