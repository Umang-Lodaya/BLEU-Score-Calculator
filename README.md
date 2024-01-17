# BLEU Score Calculator

## Overview
BLEU Score Calculator is a Python tool built with Streamlit for evaluating the quality of machine-generated text using the BLEU (Bilingual Evaluation Understudy) metric. This application allows users to calculate the BLEU score for a given prediction text and reference text(s), providing insights into the accuracy of machine-generated content.

## Features

- **BLEU Score Calculation:** Evaluate the quality of machine-generated text using BLEU scores.
- **Customizable n-gram Evaluation:** Adjust the n-gram size for precision calculation.
- **Interactive Streamlit Web Application:** User-friendly interface for easy input and result visualization.

## Table of Contents

- [How to Use](#how-to-use)
- [Installation](#installation)
- [Usage](#usage)

## How to Use

1. Enter the prediction text in the "Prediction Text" input box.
2. Enter the reference text(s) in the "Reference Text(s)" input box. Each reference should be on a separate line.
3. Click the "Start Calculating" button.
4. View the BLEU score and detailed information about the calculation, including Clipped Precision, Global Average Precision, Brevity Penalty, and the final BLEU Score.

## Installation

Install the required dependencies using the following command: ```pip install requirements.txt```

## Usage

Run the Streamlit application with the following command: ```streamlit run app.py```
