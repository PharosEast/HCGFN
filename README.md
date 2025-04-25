# Heterogeneous Contrastive Graph Fusion Network for Classification of Hyperspectral and LiDAR Data
Haoyu Jing, Sensen Wu, Laifu Zhang, Fanen Meng, Yiming Yan, Yuanyuan Wang, Zhenhong Du



The code in this toolbox implements the Heterogeneous Contrastive Graph Fusion Network for Classification of Hyperspectral and LiDAR Data (Submitting to TGRS)

## Abstract

![](./assests/pipeline.png)

In recent years, the rapid advancement of multisensory platforms has significantly increased the availability of multisource remote sensing data, facilitating its systematic application to various tasks. The joint classification of hyperspectral images (HSIs) and light detection and ranging (LiDAR) data remains a critical research topic, with a key challenge being the effective extraction and integration of complementary information from multi-source remote sensing data. However, existing graph convolutional networks (GCNs)-based methods often fail to account for the heterogeneous topological relationships between HSI and LiDAR. Moreover, the discriminative power of HSI and LiDAR features extracted by existing methods is insufficient. In addition, existing methods are unable to fully exploit the rich self-supervised information present in local neighborhood. To address these limitations, we propose a heterogeneous contrastive graph fusion network (HCGFN) for the joint classification of HSI and LiDAR data. First, we propose a branch enhancement module to enhance the discriminative power of HSI and LiDAR. Second, a contrastive learning module is introduced to effectively leverage the rich self-supervised information present in local neighborhood. Finally, we propose a dynamic heterogeneous graph structure learning module to model heterogeneous relationship and achieve efficient interaction and effective fusion between HSI and LiDAR. The extensive experimental results on three benchmark datasets indicate the effectiveness of the proposed HCGFN compared with other state-of-the-art methods. Specifically, under limited training samples, the proposed HCGFN outperformed state-of-the-art methods in overall accuracy by 5.10%, 2.46%, and 8.79% on datasets Trento, MUUFL, and Houston2013, respectively. 

## News

- **The core code of HCGFN is released.**
- ***We will release the complete code within one month after the formal publication.***

System-specific notes
---------------------

Our model is implemented by the open-source PyTorch 1.13.0 framework with Python 3.8.10.

Contact Information
--------------------

Haoyu Jing: 12138033@zju.edu.cn<br>
Zhejiang Provincial Key Laboratory of GIS, School of Earth Sciences, Zhejiang University, Hangzhou 310058.

## License

This project is open sourced under GNU General Public License v3.0.
