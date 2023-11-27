# Table of Contents
1. [Project Overview](#mlops-for-document-classification-models)
2. [Team Members](#team-members)
3. [Project Directory Structure](#project-directory-structure)
4. [Data Preparation](#data-preparation)
   - [Data Collection](#1-data-collection)
   - [Data Processing](#2-data-processing)
5. [Model Training](#model-training)
   - [Machine Learning Framework](#1-machine-learning-framework)
   - [Model Evaluation](#2-model-evaluation)
6. [Model Packaging with BentoML](#model-packaging-with-bentoml)
7. [Configuring BentoML for Data Serialization](#configuring-bentoml-for-data-serialization)
8. [Model Serving](#model-serving)
9. [Model Testing](#model-testing)
10. [Documentation and Presentation](#documentation-and-presentation)
11. [Suggested Tech Stack](#suggested-tech-stack)
12. [Repository Structure](#repository-structure)
13. [Open Issues and Milestones](#open-issues-and-milestones)
14. [Presentations](#presentations)
15. [Future Work](#future-work)


# MLOps for Document Classification Models

This project demonstrates a workflow for creating, training, packaging, and serving a document classification model, focusing on the 20 Newsgroups dataset. It incorporates machine learning and natural language processing techniques.

## Team Members
- Hogea Eduard
- Iasmina Popovici
- Diana Călina

## Project Directory Structure
Our project is organized into distinct directories, each dedicated to a specific aspect of our work. This structure enhances the project's readability and accessibility.

### Directory Layout and Contents:

#### 1. Data Exploration, Visualization, and Preprocessing
This directory contains scripts and notebooks for initial data handling, including fetching, preprocessing, and exploratory analysis.
- `fetch_dataset.py`: Script for downloading and loading the dataset.
- `advanced_preprocess.py`: Advanced data preprocessing functions.
- `exploration.ipynb`: Jupyter notebook for data exploration and visualization.

#### 2. Feature Engineering
Dedicated to scripts that focus on extracting and handling features from the data.
- `feature_extraction.py`: Script for extracting and managing features for model training.

#### 3. Model Development
Houses scripts related to the development and testing of machine learning models.
- `classification_model.py`: Script for building and training the classification model.
- `zero_shot_transformer.py`: Script for implementing and experimenting with a zero-shot learning model.


#### 4. Images
Contains generated images, such as model outputs and visualizations.
- Images related to model evaluation like confusion matrices are stored here for reference and documentation purposes.


## Data Preparation

### 1. Data Collection
- **Dataset**: 20 Newsgroups dataset, including text documents and their corresponding labels.

### 2. Data Processing
- **Tools**: NLTK for text preprocessing (tokenization, removing stop words, lemmatization).
- **Current State**: Basic text preprocessing implemented in `advanced_preprocess.py`.

## Model Training

### 1. Machine Learning Framework
**Method**: Employ scikit-learn for training various classifiers for document classification.

**Models**:
* Multinomial Naive Bayes
* Logistic Regression
* Linear Support Vector Machine (SVM)

**Current State**: Model training for all three classifiers is implemented in classification_model.py. Each model is evaluated on the 20 Newsgroups dataset.

### 2. Model Evaluation
**Implementation**: The performance of each model is evaluated using two key methods.
The evaluation metrics are implemented and displayed in the classification_model.py script, with results printed to the console and confusion matrices saved as PNG files.

1. **Confusion Matrix**: Visual representation of the performance of each classifier. Separate confusion matrices are generated and saved for each model, providing insights into the true positives, false positives, false negatives, and true negatives for each class.

2. **Classification Report**: Includes precision, recall, f1-score, and support for each class. This report is generated for each classifier, providing a detailed view of their performance across different categories.

**Results**:
- **Naive Bayes**: Demonstrated moderate performance with some variance across different categories.
- **Logistic Regression**: Showed improved performance over Naive Bayes, with higher precision and recall in most categories.
- **Linear SVM**: Exhibited the best performance among the three, with high accuracy and balanced precision and recall across various classes.

## Model Packaging with BentoML
- **Note**: Future scope for packaging the model using BentoML.

## Configuring BentoML for Data Serialization
- **Note**: Future scope for configuring BentoML to handle JSON and Multi-Part serialization.

## Model Serving
- **Note**: Future scope for setting up a BentoML server and creating an API endpoint.

## Model Testing
- **Note**: Future scope for developing a Python client for model testing and verification.

## Documentation and Presentation
- **Current State**: Basic documentation provided. Comprehensive documentation and presentation are part of future work.

## Suggested Tech Stack
- **Programming Language**: Python
- **NLP Libraries**: NLTK (current), spaCy (future)
- **ML Libraries**: scikit-learn
- **Model Packaging and Serving**: BentoML (future)
- **Client for Testing**: Python client (future)


## Open Issues and Milestones
- **Issue 9**: Development of a Python Client for Document Handling and API Interaction. Milestone 3 (Deadline: 09.01.2024).
- **Issue 8**: Preparation for Upcoming Presentation on 28.11. Milestone 1 (Deadline: 28.11.2023).
- **Issue 7**: GitHub Repository Organization and Issue Management. Milestone 1 (Deadline: 28.11.2023).
- **Issue 6**: Update and Refine README Documentation. Milestone 1 (Deadline: 28.11.2023).
- **Issue 5**: Using BentoML for Testing Different Linear Models. Milestone 2 (Deadline: 12.12.2023).
- **Issue 4**: Integration of BentoML for API Endpoint for JSON Documents. Milestone 3 (Deadline: 09.01.2024).
- **Issue 3**: Development of Comprehensive Unit Tests. Milestone 3 (Deadline: 09.01.2024).
- **Issue 2**: Integration of BentoML for Server and Data Serialization. Milestone 2 (Deadline: 12.12.2023).
- **Issue 1**: Integrate Zero Shot Classifier with Transformer. Milestone 2 (Deadline: 12.12.2023).

## Presentations
- **Milestone 1 Presentation** (Deadline: 28.11.2023): [View Presentation](https://docs.google.com/presentation/d/1ViKHCMRyTVjxCQlsLhDuiJC7XtjG7_u1FhZ6M06_4JU/edit?usp=sharing)
- **Milestone 2 Presentation** (Deadline: 12.12.2023): *Link to be provided*
- **Milestone 3 Presentation** (Deadline: 09.01.2024): *Link to be provided*
- **Final Project Presentation**: *Link to be provided*

## Future Work
- Expand the use of NLP libraries like spaCy.
- Integrate BentoML for model packaging and serving.
- Develop a Python client for model testing.
- Enhance documentation and prepare for project presentation.

