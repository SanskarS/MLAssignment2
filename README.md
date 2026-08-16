## Problem statement
Dry beans need to be identified and classified based on 16 features(12 dimensions and 4 shape forms). These beans could be classified into 7 different types. Engineer an intelligent system that is able to identify/classify these beans based on the given features.

## Dataset description
Images of 13,611 grains of 7 different registered dry beans were taken with a high-resolution camera. A total of 16 features; 12 dimensions and 4 shape forms, were obtained from the grains.
1. Area (A): The area of a bean zone and the number of pixels within its boundaries.
2. Perimeter (P): Bean circumference is defined as the length of its border.
3. Major axis length (L): The distance between the ends of the longest line that can be drawn from a bean.
4. Minor axis length (l): The longest line that can be drawn from the bean while standing perpendicular to the main axis.
5. Aspect ratio (K): Defines the relationship between L and l.
6. Eccentricity (Ec): Eccentricity of the ellipse having the same moments as the region.
7. Convex area (C): Number of pixels in the smallest convex polygon that can contain the area of a bean seed.
8. Equivalent diameter (Ed): The diameter of a circle having the same area as a bean seed area.
9. Extent (Ex): The ratio of the pixels in the bounding box to the bean area.
10. Solidity (S): Also known as convexity. The ratio of the pixels in the convex shell to those found in beans.
11. Roundness (R): Calculated with the following formula: (4piA)/(P^2)
12. Compactness (CO): Measures the roundness of an object: Ed/L
13. ShapeFactor1 (SF1)
14. ShapeFactor2 (SF2)
15. ShapeFactor3 (SF3)
16. ShapeFactor4 (SF4)
17. Class (Seker, Barbunya, Bombay, Cali, Dermosan, Horoz and Sira)

(Data set is obtained from UCIML repository, dataset id=602)

#### GitHub Repository Link:- [https://github.com/SanskarS/MLAssignment2](https://github.com/SanskarS/MLAssignment2)

## Models used:


| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.9212 | 0.9945 | 0.9344 | 0.9335 | 0.9339 | 0.9047 |
| Decision Tree | 0.8974 | 0.9479 | 0.9135 | 0.9139 | 0.9136 | 0.8759 |
| kNN | 0.918 | 0.9848 | 0.9337 | 0.9308 | 0.9322 | 0.9007 |
| Naive Bayes(Gaussian) | 0.8924 | 0.9912 | 0.9023 | 0.9035 | 0.9024 | 0.8702 |
| Naive Bayes(Multinomial) | 0.5278 | 0.9601 | 0.6168 | 0.441 | 0.4424 | 0.4285 |
| Random Forest (Ensemble) | 0.9198 | 0.9939 | 0.933 | 0.9317 | 0.9323 | 0.9029 |


| ML Model Name | Observation about model performance                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| --- |----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Logistic Regression | Best overall performer. Near to perfect AUC with highest accuracy and F1. Good, robust baseline, surprisingly hard to beat even with advanced models.                                                                                                                                                                                                                                                                                                                                                                            |
| Decision Tree | Simplest and most interpretable, but lags at AUC. Prone to overfitting and variance.                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| kNN | Good accuracy and AUC, nearly matching Logistic Regression. No training cost, but O(n) inference and memory may blow up on large datasets.                                                                                                                                                                                                                                                                                                                                                                                       |
| Naive Bayes(Gaussian) | Decent accuracy, remarkably high AUC (0.9912).Strong class ranking but weaker hard thresholds.                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| Naive Bayes(Multinomial) | This Model is showing misfit. Accuracy crashes to 0.53 despite 0.96 AUC. Continuous features break it.                                                                                                                                                                                                                                                                                                                                                                                                                           |
| Random Forest (Ensemble) | Ensemble generalizes well, near to perfect accuracy and AUC. Bagging reduces variance over single trees. Marginally below Logistic Regression, slightly costlier training.                                                                                                                                                                                                                                                                                                                                                       |
| Overall Winner for your Dataset?  | Logistic Regression is the overall winner. <br>- Highest Accuracy (0.9212), F1 (0.9339), and MCC (0.9047) <br>- Near-perfect AUC (0.9945) — second only to Random Forest's 0.9939? No, LR's 0.9945 is actually the highest <br>- Simplest, most efficient model in the group — no hyperparameter tuning, fast training/inference, easily interpretable <br>Random Forest comes closest (0.9939 AUC), but LR edges it on every metric while being far cheaper. For production, LR is the clear pick; RF is the sensible fallback. |

