Abstract


Intro
- problem definition
- Santander dataset for banking
- project goal
  - competition
  - want to maximize ranking -> prediction on provided test set


Description of the Data 
- Describe the problem in depth
  - Train/test split
- what does the data look like
- what are interesting things about the data (initial)
  - anonymized
- kaggle format and submission process


Literature Review
- Papers referencing techniques for binary classification
- What are naive bayes, logistic regression, xgboost, lgbm algorithms
- 3 papers?
  - XGBoost/LGBM/Gradient Boosting Trees
  - Data Imbalance
    - https://www3.nd.edu/~dial/publications/chawla2004editorial.pdf
  - Bayesian Cross Validation for Hyper-parameter Tuning
  - OPTIONAL: specific binary classification solution


Proposed Methodology - Strategy/Methodology (What was our plan)
%% high level description of what techniques/methods are attempted in the report %%
- describing why NB or logistic regression could apply to this problem
- Bayesian Cross Validation
- Describe steps of the flow chart
  - Data cleaning
  - EDA
  - feature extraction
    - Get insight on data (is it normally distributed? etc)
  - Classification
    - Initial algorithms to benchmark performance
  - Iteration and refinement to improve results
- Definitions of techniques


Preprocessing And Exploratory Data Analysis
    Cleaning Methods
      - Null values
      - normalization
      - outliers (boxplots: put in appendix)
    
    Identifying opportunity for dimensionality reduction
      - What is the benefit and reason behind performing correlation -> PCA
      - Correlation Heatmap plot description of results
      - PCA description of principle components and results
    
    Comparing Training and Test set provided by Competition
      - Why do we need to verify similar distribution
      - How we decided to test/identify
      - Describe results from histograms
      - Conclusion about two sets and results from anecdotal submissions
    
    Challenges in the Dataset
      - After all the above what are going to be the issues with the dataset



Implementation and Setup
  - what packages did we use
  - how did we program the models, which tools
  - Talk about script to tune hyper parameters
  - Bayesian Cross Validation
  - Focus on tools used and environment setup
   
results and discussion
    Initial 4 models
      - Naive Bayes
        - Based on the results in section above, determined NB would be viable
        - 
      - Log regression
      - XGBoost
      - LGBM
    
    Ensembling the Four models
      - voting
      - averaging
      - mention shortcomings of the approach
    
    Super Tuned LGBM model (.89 score)
      - results from hyper parameter tuning

abstract, introduction, literature review, problem definition, methods and approach, results and discussion, conclusion and reference

