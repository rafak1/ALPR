# ALPR
Automatic License Plate Recognition

In this project we will compare models for license plate detection and recognition.

## Methods
1. **YOLOv5**
2. **Filtering**
3. **Tesseract OCR**
4. **CRNN**
5. **CNN**
6. **Hog + SVM**

## Setup
The required libraries are listed in the `requirements.txt` file. You can install them using pip:

```bash
pip install -r requirements.txt
```

To run tesseract, it needs to be installed first.

## CRNN
The CRNN model was trained on the dataset https://www.kaggle.com/datasets/abdelhamidzakaria/european-license-plates-dataset?resource=download.

The model, scripts are located in the **OCR/CRNN** directory.s
A model training / usage script is provided in the **run_crnn.py** file.

Then a showcase of all the methods are presented in a jupyter notebook **notebook.ipynb**.

Project created by:
- Rafał Kajca
- Michał Miziołek