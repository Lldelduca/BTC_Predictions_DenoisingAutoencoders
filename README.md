# Enhancing Bitcoin Price Predictions: The Role of Denoising Autoencoders for Noise Reduction and Feature Selection
_Bachelor Thesis by Luca Leimbeck_

This project introduces a hybrid framework combining Denoising Autoencoders (DAE) and machine learning models to enhance Bitcoin price predictions. By leveraging DAEs for noise reduction and feature selection, the study demonstrates improved prediction accuracy using fundamental indicators, technical indicators, and lagged prices as inputs. The complete study is accessible through the Erasmus University Rotterdam Thesis Repository.

## Abstract 
Bitcoin, a highly volatile financial asset, presents significant forecasting challenges due to
its speculative nature and dependence on various factors. This paper proposes a two-stage
framework to improve Bitcoin price prediction using three types of predictors: fundamental
indicators, technical indicators, and lagged prices. In the first stage, a Denoising Autoencoder is employed to minimize noise and reduce the subset of potential predictors. To assess
the importance of this model, its results are compared to those obtained using a traditional
Autoencoder and no feature selection. In the second stage, these predictors are input into
linear regression, Support Vector Regression, and Long Short-Term Memory models. Using data from August 2011 to May 2024, we evaluate the models based on mean absolute
error, root mean squared error, and mean absolute percentage error, against a simple moving average benchmark. Our findings show that Long Short-Term Memory models exhibit
greater performance than the other models. Furthermore, the Denoising Autoencoder significantly enhances model performance, validating its effectiveness in noise reduction and
feature selection.

## Structure

This code replicates the paper "Extracting and Composing Robust Features with Denoising Autoencoders" by using deep learning models to create denoising autoencoders (DAEs) for feature extraction and noise reduction. Additionally, it includes an extension for predicting Bitcoin (BTC) prices using various financial and blockchain indicators.

The code prepares the MNIST dataset for training and validation. It splits the data into training, validation, and test sets, and sets up data loaders for batch processing. Two neural network models are defined: the Stacked Autoencoder (SAE) and the Stacked Denoising Autoencoder (SDAE). The SAE is an encoder-decoder architecture without noise addition, where the encoder reduces the input dimensions step by step, and the decoder reconstructs the input from the reduced dimensions. The SDAE is similar to the SAE but with added noise to the input data to make the model robust to noisy inputs. A mask is applied to corrupt a portion of the input data during training. We include functions to train the models and validate them using the mean squared error loss function. The code also plots the original, noisy, and reconstructed images from the autoencoder outputs. The trained models are evaluated on the test dataset, and the average test loss is calculated. Example images, noisy images, and reconstructed images are displayed to visualize the model's performance.

The extension involves predicting Bitcoin prices using various features and a two-step approach. First, we fetch, preprocess, and split the price determinants using custom web scrapers, as well as the TA library and the Yahoo YQL Finance API. This part of the code can be found under the Data section. Next, under the Methodology section the used the definitions for the Denoising Autoencoder, Autoencoder, linear regression model, Support Vector Regression model, and Long Short-Term Memory network are given. Hyperparameter tuning is performed using Optuna. We follow up by selecting the most important (and robust) features. Finally, the extracted features, as well as the non-feature selected price determinants (full and noisy set), are used as input to construct the price predictions. The models’ predictions are compared against a simple moving average benchmark.

The code includes validation checks for the Linear Regression model, ensuring its validity using statistical tests like Durbin-Watson, White's Test, and Jarque-Bera Test. Residuals and kernel density plots are generated for further analysis.

To run the code, ensure all dependencies are installed, then execute the code cells in order. The results, including plots and evaluation metrics, will be displayed as the code runs.

## Project Flowchart

```mermaid
flowchart TD
    %% Data Sources
    subgraph "Data Acquisition"
        MNIST["MNIST Data Loader"]:::data
        Finance["Financial Data Scrapers"]:::data
    end

    %% Preprocessing
    Preproc["Splitting, Scaling, Noise Injection, Lag Features"]:::process

    %% Encoding
    subgraph "Encoding Stage"
        SAE["SAE Training"]:::model
        SDAE["SDAE Training"]:::model
    end

    %% Feature Selection
    FS["Feature Selection & Combination"]:::process

    %% Predictive Modeling
    subgraph "Predictive Modeling"
        LR["Linear Regression + Optuna"]:::model
        SVR["SVR + Optuna"]:::model
        LSTM["LSTM + Optuna"]:::model
    end

    %% Evaluation and Visualization
    Eval{"Evaluation & Statistical Tests"}:::decision
    Viz["Visualization & Reporting"]:::process

    %% Documentation
    subgraph "Project Documentation"
        README["README.md"]:::data
        LICENSE["LICENSE"]:::data
    end

    %% Data Flow
    MNIST -->|"raw images"| Preproc
    Finance -->|"financial indicators"| Preproc

    Preproc -->|"noisy images"| SAE
    Preproc -->|"noisy images"| SDAE
    Preproc -->|"raw features"| FS
    SAE -->|"feature vectors"| FS
    SDAE -->|"feature vectors"| FS

    FS -->|"features"| LR
    FS -->|"features"| SVR
    FS -->|"features"| LSTM

    LR -->|"predictions"| Eval
    SVR -->|"predictions"| Eval
    LSTM -->|"predictions"| Eval

    Eval -->|"results"| Viz

    %% Click Events
    click MNIST "https://github.com/lldelduca/btc_predictions_denoisingautoencoders/blob/main/Full_Notebook.ipynb"
    click Finance "https://github.com/lldelduca/btc_predictions_denoisingautoencoders/blob/main/Full_Notebook.ipynb"
    click Preproc "https://github.com/lldelduca/btc_predictions_denoisingautoencoders/blob/main/Full_Notebook.ipynb"
    click SAE "https://github.com/lldelduca/btc_predictions_denoisingautoencoders/blob/main/Full_Notebook.ipynb"
    click SDAE "https://github.com/lldelduca/btc_predictions_denoisingautoencoders/blob/main/Full_Notebook.ipynb"
    click FS "https://github.com/lldelduca/btc_predictions_denoisingautoencoders/blob/main/Full_Notebook.ipynb"
    click LR "https://github.com/lldelduca/btc_predictions_denoisingautoencoders/blob/main/Full_Notebook.ipynb"
    click SVR "https://github.com/lldelduca/btc_predictions_denoisingautoencoders/blob/main/Full_Notebook.ipynb"
    click LSTM "https://github.com/lldelduca/btc_predictions_denoisingautoencoders/blob/main/Full_Notebook.ipynb"
    click Eval "https://github.com/lldelduca/btc_predictions_denoisingautoencoders/blob/main/Full_Notebook.ipynb"
    click Viz "https://github.com/lldelduca/btc_predictions_denoisingautoencoders/blob/main/Full_Notebook.ipynb"
    click README "https://github.com/lldelduca/btc_predictions_denoisingautoencoders/blob/main/README.md"
    click LICENSE "https://github.com/lldelduca/btc_predictions_denoisingautoencoders/tree/main/LICENSE"

    %% Styles
    classDef data fill:#AED6F1,stroke:#1B4F72,color:#1B2631;
    classDef process fill:#ABEBC6,stroke:#1D8348,color:#145A32;
    classDef model fill:#F9E79F,stroke:#B7950B,color:#7D6608;
    classDef decision fill:#F5B7B1,stroke:#943126,color:#641E16;
