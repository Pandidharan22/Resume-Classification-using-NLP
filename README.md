# Resume Classification using NLP

This project is designed to classify resumes into predefined categories using Natural Language Processing (NLP) and Machine Learning techniques. It demonstrates how to preprocess text data, extract relevant features, and train classification models to predict the domain or field of a candidate's resume.

## Project Objective

To build an automated system that can accurately categorize resumes into various job-related fields such as Data Science, HR, DevOps, Testing, Web Development, etc., based on the text content in resumes.

## Project Pipeline

1. **Data Loading**  
   Load the dataset containing resumes and their corresponding categories.

2. **Exploratory Data Analysis (EDA)**  
   Understand the structure of the dataset, check for class distribution and imbalances.

3. **Text Preprocessing**  
   - Tokenization  
   - Lowercasing  
   - Removing stopwords, punctuation, and special characters  
   - Lemmatization

4. **Feature Engineering**  
   - TF-IDF Vectorization  
   - Word Cloud Generation

5. **Model Building**  
   Train multiple machine learning models:
   - Logistic Regression
   - Random Forest Classifier
   - Support Vector Machine (SVM)
   - Multinomial Naive Bayes

6. **Model Evaluation**  
   Evaluate models using metrics like accuracy, precision, recall, and F1-score.

7. **Prediction and Inference**  
   Predict the category of a resume based on user input.

## Dataset

The dataset contains resumes and their labels. Each entry consists of:
- 'Category': The class or job domain the resume belongs to.
- 'Resume': The raw text content of the resume.

You can download the dataset used in this project from [here](https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset).

## Requirements

Install the required packages using the following command:

```bash
pip install -r requirements.txt
```

## Folder Structure
```bash
Resume_Classification/
│
├── Resume_Classification.ipynb    # Main notebook
├── requirements.txt               # Python package requirements
├── dataset/                       # Contains resume data (if applicable)
├── README.md                      # Project documentation
```

## How to Run

1. **Clone the repository**:
```bash
git clone https://github.com/your-username/resume-classification.git
cd resume-classification
```
2. **Install dependencies**:

```bash
pip install -r requirements.txt
```
3. **Open the Jupyter notebook**:

```bash
jupyter notebook Resume_Classification.ipynb
```

## Future Improvements
   - Deployment via Streamlit or Flask for real-time classification
   - Integration with OCR for PDF/DOCX resume uploads
   - Incorporate deep learning (e.g., BERT, LSTM) for better accuracy

## License
This project is open-source and available under the MIT License.

## Author
Pandidharan
