# PMLTermProject
Probabilistic Machine Learning Project Spring 2024


This project was made with the intention of being able to determine if a patient is at risk of a lethal outcome from a COVID-19 infection. This was attempted by analyzing the CT scans of patients alongside their patient records using a variational auto-encoder (VAE).

The main structure of the VAE comes from the main TensorFlow example of a VAE "https://www.tensorflow.org/tutorials/generative/cvae". This project first focused on optimizing its reconstruction ability by exploring how changing the latent space as well as amount of filters used in the convolutional layers affected performance. 

The idea behind the project was to train the VAE to only construct those CT scans of those with an optimal outcome ('discharged') and hence would be less likely to accurately recreate CT scans of those with non-optimal outcomes ('decesead'). This would in turn allow someone to find those with harmful indicators found by the VAE within a persons CT scan by finding a larger amount of discrepencies between the origional of an image and one that has been reconstructed by the VAE.

For this project the Stony Brook University COVID-19 provided by the Cancer Imaging Archive (citation below) was used.

-To download this set the NBIA data retriever must be used 

-DICOM files have to be processed into images using the python file 
PreProcessingFromNBIADataREtriever.py within the repository

-The images must be pre processed for each of the tests using the PreProcessingImagesForVAE128.py and PreProcessingImagesForVAE128HealthyVsNot.py 

-The variable optimization testing is then done on VAEFilerTesting.ipynb, VAELatentSpaceTEsting.ipynb

-The testing for differentiating between patient outcomes is done in DischargedVsDeceasedTesting.ipynb

-Final analysis is done in the MATLAB file AnalysisMATLABFile

	
CITATION FOR DATA SET
Saltz, J., Saltz, M., Prasanna, P., Moffitt, R., Hajagos, J., Bremer, E., Balsamo, J., & Kurc, T. (2021). Stony Brook University COVID-19 Positive Cases [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/TCIA.BBAG-2923
