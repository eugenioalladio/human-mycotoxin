# Regression ####
library(tidyr) # data editing
library(mlr) # ML computation
library(ggplot2) # plotting
library(gridExtra) # plotting

# Import the data
data <- read.csv("regression_data.csv",sep=";",header=TRUE)

# Remove volunteers ID
data_regr = data[,c("Dose","BMI","DON","DON.3.GlcA","DON.15.GlcA")]

# Define the regression task selecting the 'Dose' column as the response continuous feature
regr.task = makeRegrTask(data = data_regr, target = "Dose")
regr.task

# Define the cross-validation strategies performing 10 iterations 
rdesc = makeResampleDesc("CV", iters = 10L)

# Select the performance measure (rmse) to be evaluated during the benchmark analysis
meas = rmse

# Define the ML models to be employed for the regression benchmark
lrns = list(makeLearner("regr.lm"), # Multiple Linear Regression
            makeLearner("regr.pcr"), # Principal Component Regression
            makeLearner("regr.plsr"), # Partial Least Squares Regression
            makeLearner("regr.kknn"), # k-Nearest-Neighbors
            makeLearner("regr.svm"), # Support Vector Machine
            makeLearner("regr.nnet"), # Neural Networks
            makeLearner("regr.randomForest"), # Random Forest  
            makeLearner("regr.xgboost")) # eXtreme Gradient Boosting

# Set-up of the benchmark analysis
bmr = benchmark(learners = lrns, tasks = regr.task, resamplings = rdesc, 
                measures = rmse, 
                show.info = TRUE)

# Name the ML models
models <- c("regr.lm","regr.pcr","regr.plsr","regr.kknn","regr.svm","regr.nnet",
            "regr.randomForest","regr.xgboost")
names.models <- c("MLR","PCR","PLS-R","k-NN","SVM","ANN","RF","XGBoost")

# Prepare a table containing the benchmark results
table <- list()
for (i in models){
  table[[i]] <- as.data.frame(bmr[["results"]][["data_regr"]][[i]][["pred"]]$data)
}

for (j in 1:length(names.models)) {
  table[[j]]["Model"] <- rep(names.models[j],nrow(table[[j]]))
}
mytable <- data.table::rbindlist(table)
colnames(mytable)[6] <- "models"
mytable$models <- as.factor(mytable$models)
mytable$models <- factor(mytable$models, levels=c("MLR","PCR","PLS-R","k-NN", "SVM", "ANN", "RF", "XGBoost"))

# Plot the benchmark results
my.formula <- y ~ x
ggplot(mytable, aes(truth, response)) +
  facet_wrap(. ~ models, ncol = 4) +
  geom_point() +
  theme_bw()+
  stat_summary(fun.data= mean_cl_normal) + 
  geom_smooth(method='lm', se = FALSE)+
  ggpmisc::stat_poly_eq(formula = my.formula, 
                        aes(label = paste(..eq.label.., ..rr.label.., sep = "~~~")), 
                        parse = TRUE) +
  labs(title = "Machine Learning regression benchmark",
       y = "Predicted dosage (nmol)", x = "Effective dosage (nmol)")

