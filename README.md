# ShuttleFlow: Learning the Distribution of Subsequent Badminton Shots using Normalizing Flows

This repo is the implementations of [ShuttleFlow: Learning the Distribution of Subsequent Badminton Shots using Normalizing Flows](https://github.com/sophialien/DifferentiableOfflineRL/blob/main/%5Bpaper%5D%20Enhancing%20Value%20Function%20Estimation.pdf). Machine Learning, Springer Nature (Special Issue for Asian Conference on Machine Learning, ACML) 2024

Authors: Yun-Hsuan Lien, Chia-Tung Lian, Yu-Shuen Wang

## Abstract
This paper introduces ShuttleFlow, a simple yet effective model designed to forecast badminton shot types and shuttle positions. This tool could be invaluable for coaches, enabling them to identify opponents' weaknesses and devise effective strategies accordingly. Given the inherent unpredictability of player behaviors, our model leverages conditional normalizing flow to generate the distributions of shot types and shuttle positions. This is achieved by considering the players and their preceding shots on the court. To augment the performance of our model, especially in predicting outcomes for players who have not previously competed against each other, we incorporate a novel regularization term. Additionally, we utilize Poisson disk sampling to reduce sample redundancy when generating the distributions. Compared to state-of-the-art techniques, our results underscore ShuttleFlow's effectiveness in forecasting shot types and shuttle positions.

<p align="center">
  <img src="https://github.com/sophialien/ShuttleFlow/blob/main/HitDistribution.png" width="700" />
</p>

---
If you find this work useful in your research, please consider citing:
```
@inproceedings{lien2024shuttleflow,
 author={Yun-Hsuan Lien, Chia-Tung Lian, Yu-Shuen Wang},
 booktitle={Machine Learning, Springer Nature (Special Issue for Asian Conference on Machine Learning, ACML)},
 year={2024}
 }
```

## Contact Information
If you have any questions, please contact Sophia Yun-Hsuan Lien: sophia.yh.lien@gmail.com
