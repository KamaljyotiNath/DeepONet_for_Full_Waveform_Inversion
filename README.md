# DeepONet for Full Waveform Inversion

This repository contains DeepONet codes for the 'Leveraging Deep Operator Networks (DeepONet) for Acoustic Full Waveform Inversion (FWI)'

## Abstract:
<p style='text-align: justify;'>
Full Waveform Inversion (FWI) is an important geophysical technique considered in subsurface property prediction. It solves the inverse problem of predicting high-resolution Earth interior models from seismic data. Traditional FWI methods are computationally demanding. Inverse problems in geophysics often face challenges of non-uniqueness due to limited data, as data are often collected only on the surface. In this study, we introduce a novel methodology that leverages Deep Operator Networks (DeepONet) to attempt to improve both the efficiency and accuracy of FWI. The proposed DeepONet methodology inverts seismic waveforms for the subsurface velocity field. This approach is able to capture some key features of the subsurface velocity field. We have shown that the architecture can be applied to noisy seismic data with an accuracy that is better than some other machine learning methods. We also test our proposed method with out-of-distribution prediction for different velocity models. The proposed DeepONet shows comparable and better accuracy in some velocity models than some other machine learning methods. To improve the FWI workflow, we propose using the DeepONet output as a starting model for conventional FWI  and that it may improve FWI performance. While we have only shown that DeepONet facilitates faster convergence than starting with a homogeneous velocity field, it may have some benefits compared to other approaches to constructing starting models. This integration of DeepONet into FWI may accelerate the inversion process and may also enhance its robustness and reliability.
</p>

The code is written in Tensorflow-2

Dataset:
The training and testing datasets considered can be downloaded from [OpenFWI](https://arxiv.org/abs/2111.02926).

The trained weights and biases can be downloaded from this [DropBox](https://www.dropbox.com/scl/fo/toielc1m50ck07azuiamz/AEPkdont5pGml6WBKr_ZKsE?rlkey=paz3hp17eiuu0a1cs42hjg2os&st=osv34tkx&dl=0).
