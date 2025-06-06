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
    subgraph "Data Sources"
        FundInd["Fundamental Indicators (e.g. macro, FX, blockchain)"]:::data
        TechInd["Technical Indicators (e.g. moving avg, RSI)"]:::data
        LagPrices["Lagged BTC Prices (60-day window)"]:::data
    end

    %% Preprocessing
    Preproc["Standardization + Interpolation + Splitting"]:::process

    %% Feature Selection
    subgraph "Feature Selection"
        NoneFS["No Selection"]:::model
        AE["Autoencoder (AE)"]:::model
        DAE["Denoising Autoencoder (DAE)"]:::model
    end

    %% Predictive Modeling
    subgraph "Modeling (Bayesian-tuned)"
        LR["Linear Regression"]:::model
        SVR["Support Vector Regression"]:::model
        LSTM["LSTM Network"]:::model
        SMA["2-Day Moving Average (Benchmark)"]:::benchmark
    end

    %% Evaluation
    Eval["Evaluation: MAE, RMSE, MAPE"]:::decision
    Results["Results: Interval-wise Performance"]:::process

    %% Documentation
    README["README.md"]:::data
    LICENSE["LICENSE"]:::data

    %% Flow
    FundInd --> Preproc
    TechInd --> Preproc
    LagPrices --> Preproc

    Preproc --> NoneFS
    Preproc --> AE
    Preproc --> DAE

    NoneFS --> LR
    AE --> LR
    DAE --> LR

    NoneFS --> SVR
    AE --> SVR
    DAE --> SVR

    NoneFS --> LSTM
    AE --> LSTM
    DAE --> LSTM

    SMA --> Eval
    LR --> Eval
    SVR --> Eval
    LSTM --> Eval

    Eval --> Results

    %% Styles
    classDef data fill:#AED6F1,stroke:#1B4F72,color:#1B2631;
    classDef process fill:#ABEBC6,stroke:#1D8348,color:#145A32;
    classDef model fill:#F9E79F,stroke:#B7950B,color:#7D6608;
    classDef decision fill:#F5B7B1,stroke:#943126,color:#641E16;
    classDef benchmark fill:#D7BDE2,stroke:#7D3C98,color:#512E5F;
