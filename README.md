# Fair Federated Learning under Domain Skew with Local Consistency and Domain Diversity

> Fair Federated Learning under Domain Skew with Local Consistency and Domain Diversity,            
> Yuhang Chen, Wenke Huang, Mang Ye
> *CVPR, 2024*

## News
* [2024-05-28] Paper has been released. [arXiv](https://arxiv.org/abs/2405.16585v1)
* [2024-03-25] Code has been released. [Digits](https://drive.google.com/drive/folders/1SSv9dqQPBGyHS3rSwoFKmpBIeF4GX-i6?usp=sharing)


## Abstract
Federated learning (FL) has emerged as a new paradigm for privacy-preserving collaborative training. Under domain skew, the current FL approaches are biased and face two fairness problems. 1) Parameter Update Conflict: data disparity among clients leads to varying parameter importance and inconsistent update directions. These two disparities cause important parameters to potentially be overwhelmed by unimportant ones of dominant updates. It consequently results in significant performance decreases for lower-performing clients. 2) Model Aggregation Bias: existing FL approaches introduce unfair weight allocation and neglect domain diversity. It leads to biased model convergence objective and distinct performance among domains. We discover a pronounced directional update consistency in Federated Learning and propose a novel framework to tackle above issues. First, leveraging the discovered characteristic, we selectively discard unimportant parameter updates to prevent updates from clients with lower performance overwhelmed by unimportant parameters, resulting in fairer generalization performance. Second, we propose a fair aggregation objective to prevent global model bias towards some domains, ensuring that the global model continuously aligns with an unbiased model. The proposed method is generic and can be combined with other existing FL methods to enhance fairness. Comprehensive experiments on Digits and Office-Caltech demonstrate the high fairness and performance of our method. 

## Citation
```
@inproceedings{FedHEAL_CVPR2024,
    author    = {Chen, Yuhang and Huang, Wenke and Ye, Mang},
    title     = {Fair Federated Learning under Domain Skew with Local Consistency and Domain Diversity},
    booktitle = {CVPR},
    year      = {2024}
}
```
