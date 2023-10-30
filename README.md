# Emotion Detector Web Application

- [Description](#description)
- [Demo](#demo)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Application](#running-the-application)
- [Usage](#usage)
  - [Live Emotion Detection](#live-emotion-detection)
  - [Emotion Detection from Photos](#emotion-detection-from-photos)
- [Model Folder](#model-folder)
- [Current Production Model](#current-production-model)
- [Contact](#contact)


## Description

The Emotion Detector Web Application leverages Facial Expression Recognition (FER) to analyze emotions in real-time and from photos. This intuitive application provides instant and accurate emotional analysis, enhancing your understanding of human responses in different contexts.

## Demo

![Emotion Detector Demo](demo.gif)

## Getting Started

These instructions will guide you through setting up and running the application on your local machine for development and testing purposes.

### Prerequisites

- Python 3.6 or higher
- pip (Python package installer)
- Virtual environment 

### Installation

1. Clone the repository:
   
   `git clone https://github.com/<USERNAME>/emotion_detector_project.git`
   
2. Navigate to the folder repository
   
   `cd engineer-diploma`

### Running the Application

Execute the bash script in the downloaded repo folder to set up the environment, install the necessary libraries, and start the Django web application:

`./run.sh`

The application will be accessible at `http://127.0.0.1:8000/` or the URL provided in your terminal.

## Usage

### Live Emotion Detection

To utilize the live emotion detection feature:

1. Open your web browser and navigate to `http://127.0.0.1:8000`.
2. Go to the "Live Detection" page.
3. Allow the necessary permissions to access your webcam.
4. Observe the real-time emotional analysis as the application identifies faces and their associated emotions.

### Emotion Detection from Photos

To perform emotion analysis on photos:

1. Open your web browser and go to `http://127.0.0.1:8000`.
2. Go to the "Photo Detection" page.
3. Upload a photo using the provided interface.
4. View the results as the application detects faces and their associated emotions in the uploaded photo.

## Model Folder

In the `model/models_code` folder, you will find six subfolders rigorously tested FER approaches. Each subfolder comes with model performance metrics, aiding you in selecting the most suitable one for your needs and code of letting to train such network. Models were trained using the merged databases (Ck+, FER2013, RAF-DB). The models include:

1. **Deep Convolutional Network**: A deep convolutional network tailored for FER.
2. **VGG16 Transfer Learning**: Employing the VGG16 architecture with pretrained weights.
3. **ConvNext Vision Transformer**: Leveraging the ConvNext architecture with pretrained weights for a unique combination of convolutional layers and transformers.
4. **ViT Vision Transformer**: Utilizing the Vision Transformer architecture with pretrained weights.
5. **BEIT Vision Transformer**: Based on the BERT pre-training technique adapted for images, with pretrained weights.
6. **Ensemble Model**: A combination of various models to enhance performance, including models with transfer learning.

Performance metrics for each model are provided to ensure you have all the necessary information to make an informed decision.

## Current Production Model

The `ConvNext Vision Transformer` model, equipped with transfer learning, is currently deployed in production due to its superior performance compared to the other models. This unique combination of convolutional layers and transformers provides an effective and efficient solution for real-time emotion detection.

## Contact

- Michał Kopczyński - mkopczynski24@gmail.com
- Project Link: https://github.com/kopczyn12/engineer-diploma
