---
Author: Vighnesh Nayak
Date: 19/08/2024
Topic: Background Separation
tags:
  - dip
  - cv
---
# Robust Principle Component Analysis
---
Robust Principal Component Analysis (RPCA) is a statistical method used to separate a matrix into a low-rank component and a sparse component. It is useful for handling data with outliers or noise. RPCA solves the problem:

$$ \text{minimize} \| L \|_* + \lambda \| S \|_1 $$

subject to:

$$ D = L + S $$

where $D$ is the original data matrix, $L$ is the low-rank matrix, $S$ is the sparse matrix, $\| L \|_*$ denotes the nuclear norm (sum of singular values) of $L$, and $\| S \|_1$ denotes the $\ell_1$-norm (sum of absolute values of entries) of $S$. The parameter $\lambda$ balances the trade-off between the low-rank and sparse components.

The traditional RPCA algorithm requires all the frames of video at process time. But for our application we need a modified approach which can process frames incrementally that is one by one also called as Online RPCA. We currently have found three promising research papers for the same

### [Online Robust Principal Component Analysis with Change Point Detection](Papers/Online_Robust_Principal_Component_Analysis_with_Change_Point_Detection.pdf)

The paper "Online Robust Principal Component Analysis with Change Point Detection" introduces an efficient method called Online Moving Window Robust Principal Component Analysis (OMWRPCA) designed to handle both slowly changing and abruptly changing subspaces in high-dimensional data. Unlike traditional batch algorithms, OMWRPCA can process streaming data in real-time, making it suitable for applications such as video surveillance, anomaly detection, and network monitoring. By integrating hypothesis testing, OMWRPCA can detect change points in the underlying subspace, ensuring accurate and timely identification of significant changes in the data structure. The method has demonstrated superior performance through extensive simulations and practical applications compared to existing state-of-the-art approaches [(Wei et al., 2017)](https://arxiv.org/abs/1702.05698).

### [Incremental Gradient on the Grassmannian for Foreground and Background Separation](Papers/Incremental_gradient_on_the_Grassmannian_for_online_foreground_and_background_separation_in_subsampled_video.pdf)

The paper "Incremental Gradient on the Grassmannian for Online Foreground and Background Separation in Subsampled Video" introduces GRASTA, a robust online algorithm for subspace estimation and background-foreground separation in video streams using random subsampling. Leveraging the Grassmannian geometry, GRASTA efficiently processes video frames, achieving high separation accuracy and computation speeds, as evidenced by its ability to process video at 46.3 frames per second on standard hardware. This method significantly improves computational efficiency while maintaining robust subspace learning, which is critical for real-time video surveillance and other applications where computational resources and quick processing are essential. The algorithm's design enables it to dynamically adapt to changes in the subspace, ensuring robust performance even with highly subsampled data [(He et al., 2012)](https://typeset.io/papers/incremental-gradient-on-the-grassmannian-for-online-4kx3xddloz?utm_source=chatgpt).

### [Online Tensor Robust Principal Component Analysis](Papers/Online_Tensor_Robust_Principal_Component_Analysis.pdf)

- The paper introduces an online tensor robust principal component analysis (RPCA) algorithm designed to efficiently decompose high-dimensional data into low-rank and sparse components, addressing the limitations of conventional batch processing methods.
- The proposed algorithm is particularly beneficial for real-time applications where data arrives sequentially, such as video surveillance, by utilizing recursive approaches to manage memory and computational efficiency.
- Key contributions include a fast tensor RPCA algorithm, an efficient tensor convolutional extension to the fast iterative shrinkage thresholding algorithm (FISTA), and an incremental tensor SVD method for tracking subspace changes.
- The paper outlines the mathematical foundations necessary for understanding the proposed algorithm, including tensor notation, the T-product, and tensor singular value decomposition (T-SVD).
- Key operations such as the Fourier transform and the block circulant matrix are introduced, which facilitate efficient computation in the proposed online framework.
- The T-SVD is defined as a crucial component for factorizing tensors, enabling the algorithm to effectively manage the low-rank and sparse components of the data.
- The algorithm consists of three main stages: sparse recovery, low-rank recovery, and subspace tracking, starting with an initial estimate of the low-rank background using tensor singular value thresholding (T-SVT).
- Foreground detection is framed as a sparse coding problem, utilizing a specially designed dictionary that enhances computational efficiency by focusing on the sparse component.
- The method employs FISTA for optimization, leveraging efficient gradient computation in the Fourier domain to improve processing speed and accuracy.