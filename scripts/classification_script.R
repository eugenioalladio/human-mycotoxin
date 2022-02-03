# Classification ####
library(tidyr) # data editing
library(mlr) # ML computation
library(ggplot2) # plotting
library(gridExtra) # plotting

# Import the data
data <- read.csv("classification_data.csv",sep=";",header=TRUE)

# Remove volunteers ID
data <- data[,2:5]

# Convert the 'Group' column to factor
data$Group <- as.factor(data$Group)

# Define the ML models to be employed for the classification benchmark
lrns = list(
  makeLearner("classif.knn"), # k-Nearest-Neighbors
  makeLearner("classif.lda"), # Linear Discriminant Analysis
  makeLearner("classif.plsdaCaret"), # Partial Least Squares Discriminant Analysis
  makeLearner("classif.naiveBayes"), # Naive Bayes
  makeLearner("classif.svm"), # Support Vector Machine
  makeLearner("classif.rpart"), # Decision tree
  makeLearner("classif.randomForest"), # Random Forest  
  makeLearner("classif.xgboost") # eXtreme Gradient Boosting
) 

# Define the classification task selecting the 'Group' column as the response categorical feature
data.task = makeClassifTask(id = "Mycotoxin data", data = data, target = "Group")
tasks = list(data.task)

# Define the cross-validation strategies performing 8 iterations 
rdesc = makeResampleDesc("CV", iters = 8L)

# Select the metrics to be evaluated during the benchmark analysis
meas = list(acc, mmce, tnr, tpr)

# Set-up of the benchmark analysis
benchmark_res = benchmark(lrns, tasks, measures = meas,rdesc)

# Name the ML models
models <- c("classif.knn","classif.lda","classif.plsdaCaret","classif.naiveBayes","classif.svm",
            "classif.rpart","classif.randomForest","classif.xgboost")
names.models <- c("k-NN","LDA","PLS-DA","NB","SVM","DT","RF","XGBoost")

# Prepare a table containing the benchmark results
table <- list()
for (i in models){
  table[[i]] <- benchmark_res[["results"]][["Mycotoxin data"]][[i]][["aggr"]]
}
mytable <- as.data.frame(unclass(table))
colnames(mytable) <- names.models
mytable <- as.data.frame((mytable))
Parameters <- c("Accuracy", "Mean Misclassification Error", "Specificity", "Sensitivity")
mytable<- cbind(Parameters,mytable)
mytable <- mytable %>%
  tidyr::pivot_longer(!Parameters, names_to = "Model", values_to = "Results")
mytable$Model <- as.factor(mytable$Model)
mytable$Model <- factor(mytable$Model, levels=c("k-NN", "LDA", "PLS-DA", "NB", "SVM", "DT", "RF", "XGBoost"))

# Plot the benchmark results
ggplot(mytable, 
       aes(Model, Results*100, fill=Model)) + 
  geom_bar( stat = "identity" ) + 
  facet_wrap(. ~ Parameters) + 
  xlab("Classification models comparison")  +
  theme_bw()+ 
  geom_text( aes( label = paste0( round(Results*100,0), "%" ), y = round(Results*100,0) ),
             vjust = 1.4, size = 3, color = "white" )+
  theme( axis.text.x = element_text( angle = 60,  hjust = 1 ) )+
  labs(title = "Machine Learning classification benchmark",
       y = "Metrics (%)", x = "Models")

# Further evaluations using Random Forest and k-NN models ####

# Define the number of available samples
n = getTaskSize(data.task)

# Divide the data into training (2/3) and test (1/3) sets
train.set = sample(n, size = round(2/3 * n))
test.set = setdiff(seq_len(n), train.set)

# Build a Random Forest model
rf_model = makeLearner("classif.randomForest", predict.type = "prob")
model_rf = train(rf_model, data.task, subset = train.set)

# Predict the test data using the developed Random Forest model
prediction_rf = predict(model_rf, task = data.task, subset = test.set)

# Compute and plot the ROC curve for Random Forest model
roc_data_rf = generateThreshVsPerfData(prediction_rf, measures = list(fpr, tpr, mmce))
plotROCCurves(roc_data_rf)

# Compute the AUC value for the Random Forest model
mlr::performance(prediction_rf, mlr::auc)

# Build a k-NN model
knn_model = makeLearner("classif.kknn", predict.type = "prob")
model_knn = train(knn_model, data.task, subset = train.set)

# Predict the test data using the developed k-NN model
prediction_knn = predict(model_knn, task = data.task, subset = test.set)

# Compute and plot the ROC curve for k-NN model
roc_data_knn = generateThreshVsPerfData(prediction_knn, measures = list(fpr, tpr, mmce))
plotROCCurves(roc_data_knn)

# Compute the AUC value for the k-NN model
mlr::performance(prediction_knn, mlr::auc)

# Compare and plot the ROC curves of the developed models
roc_comparison = generateThreshVsPerfData(list("Random Forest" = prediction_rf, "k-NN" = prediction_knn), measures = list(fpr, tpr))
plotROCCurves(roc_comparison)


