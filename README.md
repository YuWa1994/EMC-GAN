# EMC-GAN for condition monitoring with emerging new classes
Welcome to the GitHub repository for EMC-GAN, a Python implementation of a GAN-based method designed to address the challenges posed by Streaming Data with Emerging New Classes (SENC) in condition monitoring of rotating machinery.

## Research Manuscripts ##
We have conducted extensive research on this method, and the results have been documented in the following manuscripts:

**1. Journal Manuscript**:
* Title: Ensembled Multi-Task Generative Adversarial Network (EMT-GAN): A Deep Architecture for Classification in Streaming Data with Emerging New Classes and its Application to Condition Monitoring of Rotating Machinery
* Authors: Yu Wang, Q. Wang, S. Bernat, A. Vinogradov
* Status: Under review
* Link to Preprint: (https://www.techrxiv.org/articles/preprint/Semi-supervised_deep_architecture_for_classification_in_streaming_data_with_emerging_new_classes_application_in_condition_monitoring/21931476/1)

Note: The preprint version only contains the results for benchmark datasets.


**2. Conference Manuscript**: 
* Title: Ensembled Multi-Classification Generative Adversarial Network for Condition Monitoring in Streaming Data with Emerging New Classes
* Authors: Yu Wang, Q. Wang, A. Vinogradov
* Event: 1st Olympiad in Engineering Science – OES 2023
* Status: Presented

## Brief Introduction ##
The primary focus of this repository is the application of our proposed EMC-GAN method on two benchmark datasets, which are:
* Case Western Reserve University (CWRU) Bearing Data Center
* Konstruktions- und Antriebstechnik (KAt) - Bearing Data Center

The core implementation of the method can be found in the 'GAN_for_NEC.py' file, while the network architectures utilized in the process are available in the 'GAN_Nets.py' file.

To run the EMC-GAN method on the provided datasets, simply execute either 'run_EMCGAN_Kat.py' for the KAt dataset or 'run_EMCGAN_CWRU.py' for the CWRU dataset.

If you have any questions, suggestions or contributions, please feel free to contact me on email yuwa@ntnu.no or yuwang1994@outlook.com.

## Note: ## 
This README is based on the information available up to August 2023.


