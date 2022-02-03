# human-mycotoxin-UGent
These scripts accompany the paper: **"INSERT TITLE"**; they describe the multivariate data analysis approeaches employed for the evaluation of the collected data.

## Provided data
The **'data' repository** contains the two input files used during the study for the calculations and the benchmark evaluations of Supervised Machine Learning models, in terms of classification (*'classification_data.csv'*) and regression (*'regression_data.csv'*). 
Briefly, the collected datasets consisted of 20 volunteers (11 female and 9 male, within the age range of 21-61 years) divided into a control group (4 subjects, i.e., those volunteers taking a placebo solution) and a DON group (16 subjects, i.e., those volunteers taking DON). Further details about the collected data are availbale in the related paper entitled **'INSERT TITLE'**.
To ease the use of the scripts and the data, please download the Git folder manually.

## Getting started
Ensure the following **R-packages** (including their dependencies) are installed and loaded correctly in your environment. 
The computations were carried out using the R version 4.0.2 (2020-06-22) -- "Taking Off Again" and the RStudio version 1.2.5001. 
If the mentioned packages are available under different version, the scripts should work as well, but the features of each package's implementation may modify the results slightly.

The required R packages are:
 - *tidyr*: a tool for working and edit the data (i.e. building pivot table converting the data between long and wide formats);
 - *ggplot2*: a well-known library for the creation of straightforward graphs and plots;
 - *mlr*: a library for Machine Learning, containing a large number of classification and regression models;
 - *gridExtra*: a package to be used in combination with ggplot2 to arrange multiple grid-based plots and to prepare tables.

## Scripts
Under the **Scripts** directory, the below two scripts are assumed to exist:

 - *classification_script.R*: containing the R-script for the benchmark evaluation of Supervised Machine Learning models for classification;
 - *regression_script.R*: containing the R-script for the benchmark evaluation of Supervised Machine Learning models for regression.

## Usage
To run the code, simply source the scripts in RStudio (after specifying the working directory within the script) – or work following the procedural part of the paper step-by-step.
