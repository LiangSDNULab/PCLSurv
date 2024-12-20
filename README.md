# PCLSurv
PCLSurv: A Prototypical Contrastive Learning-based Multi- omics Data Integration Model for Cancer Survival Prediction
## PCLSurv Model
![Framework1117](https://github.com/user-attachments/assets/ce6c0ede-bc2a-4108-941b-bb73ff3c3d33)
Accurate cancer survival prediction remains a critical challenge in clinical oncology, largely due to the complex and multi-omics nature of cancer data. Existing methods often struggle to capture the comprehensive range of informative features required for precise predictions. Here, we introduce PCLSurv, an innovative deep learning framework designed for cancer survival prediction using multi-omics data. PCLSurv integrates autoencoders to extract omics-specific features and employs sample-level contrastive learning to identify distinct yet complementary characteristics across data views. Then, features are fused via a bilinear fusion module to construct a unified representation. To further enhance the model’s capacity to capture high-level semantic relationships, PCLSurv aligns similar samples with shared prototypes while separating unrelated ones via prototypical contrastive learning. As a result, PCLSurv effectively distinguishes patient groups with varying survival outcomes at different semantic similarity levels, providing a robust framework for stratifying patients based on clinical and molecular features. We conduct extensive experiments on 11 cancer datasets the comparison results confirm the superior performance of PCLSurv over existing alternatives.
## Requirements
torch==1.7.1
sklearn-pandas==2.2.0
pandas==2.0.3
numpy==1.24.3
