# Speech-Recogition:: # african_language-Speech_Recognition


## Check Assessment Presentation:
👉 

**Table of Contents**

- [african_language-Speech_Recognition](#african_language-Speech_Recognition)
  - [Overview](#overview)
  - [Scenario](#scenario)
  - [Approach](#approach)
  - [Project Structure](#project-structure)
    - [data:](#data)
    - [models](#models)
    - [notebooks](#notebooks)
    - [scripts](#scripts)
    - [tests](#tests)
    - [logs](#logs)
    - [root folder](#root-folder)
  - [Installation guide](#installation-guide)

# african_language-Speech_Recognition

African language Speech Recognition - Speech-to-Text Collaborative Project.
## Overview
This repository is used for week 4 challenge of `10Academy`. The instructions for this project can be found in the challenge document. This is a collaborative project between eight trainers at `10Academy` bootcamp:
1. `Danayt Bulom`
2. `Faith Bagire `
3. `Yonas Tadesse` 
4. `Diye Mark`
5. `Daisy Okacha `
6. `Gezahegne Wondachew`
7. `Hikma Burhan `
8. `Ammon Leulseged`


## Scenario
The World Food Program wants to deploy an intelligent form that collects nutritional information of food bought and sold at markets in two different countries in Africa - Ethiopia and Kenya. The design of this intelligent form requires selected people to install an app on their mobile phone, and whenever they buy food, they use their voice to activate the app to register the list of items they just bought in their own language. The intelligent systems in the app are expected to live to transcribe the speech-to-text and organize the information in an easy-to-process way in a database.

You work for the Tenacious data science consultancy, which is chosen to deliver speech-to-text technology for Swahili. Your responsibility is to build a deep learning model that is capable of transcribing a speech to text. The model you produce should be accurate and is robust against background noise.


## Approach
The project is divided and implemented by the following phases
- Data pre-processing
- Modelling using deep learning
- Serving predictions on a web interface
- Interpretation & Reporting

## Project Structure
The repository has a number of files including python scripts, jupyter notebooks, pdfs and text files. Here is their structure with a brief explanation.

### data:
- the folder where the dataset csv files are stored

### models:
- the folder where models' pickle files are stored

### notebooks:
- the folder where the notebooks are stored

### scripts
-All scripts are stored in the scripts folder.
### tests:
- the folder containing unit tests for components in the scripts

### logs:
- the folder containing log files (if it doesn't exist it will be created once logging starts)

### team_screenshots
- the root folder of the project
## Installation guide

git clone https://github.com/African-language-Speech-Recognition/african_language-Speech_Recognition.git
cd african_language-Speech_Recognition
pip install -r requirements.txt

## Source of The Dataset
- This repo uses public dataset provided by ALFFA_PUBLIC [ https://github.com/besacier/ALFFA_PUBLIC/tree/master/ASR/AMHARIC ]

```