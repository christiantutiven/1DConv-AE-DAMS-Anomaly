# Anomaly Detection in Dams with 1D Convolutional Siamese Neural Networks



Full implementation of a \*\*Siamese Neural Network\*\* model based on \*\*1D-CNN\*\* for anomaly detection in dam structural monitoring data. The approach:



\- Processes displacement series and simulated environmental data with FEM.

\- Temporally enriches using 7-day sliding windows.

\- Uses networks with Conv1D layers (128→256 filters) to extract temporal and inter-sensor patterns.

\- Generates positive/negative pairs to train the similarity between normal and anomalous conditions.

\- Employs early stopping and saves the model at the first validation epoch at 100%.



This repository includes a series of Jupyter notebooks that implement and evaluate an anomaly detection approach in dam structural monitoring data using 1D Siamese convolutional neural networks.



\## Description



The complete workflow includes:



1\. \*\*Exploratory Data Analysis\*\*

Exploration of time series of displacements and environmental variables to understand their distribution and main characteristics.



2\. \*\*Preprocessing and Generation of Time Windows\*\*

Normalization, segmentation into sliding windows, and creation of indices to feed the network.



3\. \*\*1D Siamese-CNN Model\*\*

Implementation of a Siamese network based on Conv1D layers to extract temporal patterns and calculate the similarity between sequence pairs (normal vs. anomalous).



4\. \*\*RoC (Reference Value) Evaluation\*\*

Calculation of classic metrics (accuracy, precision, recall, F1) and analysis of the ROC and AUC curves on unseen data.

\## Notebooks



\- `1- Exploratory Data Analysis.ipynb`  

\- `2- Preprocess indices.ipynb`  

\- `3- Siamese Neural network.ipynb`  

\- `4- Testeo-ROC.ipynb`  



\## Dependencies



\- Python >= 3.8  

\- numpy  

\- pandas  

\- matplotlib  

\- seaborn  

\- scikit-learn  

\- torch (PyTorch)  

\- tqdm  



Install them with:



```bash

pip install -r requirements.txt

```



\## Repository structure



```

├── 1- Exploratory Data Analysis.ipynb

├── 2- Preprocess indices.ipynb

├── 3- Siamese Neural network.ipynb

├── 4- Testeo-ROC.ipynb

├── README.md

└── requirements.txt

```



\## Usage



1\. Clone the repository:



```bash

git clone https://github.com/christiantutiven/1DConv-AE-DAMS-Anomaly.git

cd 1DConv-AE-DAMS-Anomaly

```



2\. Install dependencies:



```bash

pip install -r requirements.txt

```



3\. Open and run the notebooks in order:



```bash

jupyter notebook

```



\## Authors



\- \*\*Christian Tutivén\*\* – cjtutive@espol.edu.ec



\## License



This project is licensed under the MIT License.

