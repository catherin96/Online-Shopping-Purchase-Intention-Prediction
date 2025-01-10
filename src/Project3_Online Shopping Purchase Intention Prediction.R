# ------------------------------------------------------------
# Online Shopping Purchase Intention (OSPI) Prediction Project
# ------------------------------------------------------------

# Install required libraries
required_libraries <- c(
  "rsample", "modeldata", "mclust", "ISLR2", 
  "tidyverse", "caret", "lattice", "rpart.plot", 
  "readr", "lubridate", "ggplot2", "ranger"
)

for (lib in required_libraries) {
  if (!require(lib, character.only = TRUE)) {
    install.packages(lib, dependencies = TRUE)
  }
}

#Import necessary libraries
library(caret)
library(lattice)
library(rsample)
library(modeldata)
library(tidyverse)
library(readr)     
library(lubridate) 
library(rpart.plot)
library(ranger)

# Load dataset
shoppers_data <- read.csv("D://online shoppers//online_shoppers_intention.csv")

# Data Cleaning
# Convert boolean values (TRUE/FALSE) to binary values (0/1) in the "Revenue" column
shoppers_data$Revenue <- as.numeric(shoppers_data$Revenue)
# Convert boolean values (TRUE/FALSE) to binary values (0/1) in the "Weekend" column
shoppers_data$Weekend <- as.numeric(shoppers_data$Weekend)

# Convert categorical variable 'Month' to numerical codes
month_codes <- c("Jan" = 1, "Feb" = 2, "Mar" = 3, "Apr" = 4, "May" = 5, "June" = 6, 
                 "Jul" = 7, "Aug" = 8, "Sep" = 9, "Oct" = 10, "Nov" = 11, "Dec" = 12)
shoppers_data$Month <- month_codes[shoppers_data$Month]

# Explore the dataset structure and summary
str(shoppers_data)
summary(shoppers_data)

# Logistic Regression Model
logistic_model <- glm(Revenue ~ Administrative + Administrative_Duration + Informational + 
                        Informational_Duration + ProductRelated + ProductRelated_Duration + 
                        BounceRates + ExitRates + PageValues + SpecialDay + Month + 
                        OperatingSystems + Browser + Region + TrafficType + VisitorType + Weekend, 
                      data = shoppers_data, family = binomial)
summary(logistic_model)

# Check coefficients and odds
log_odds <- coef(logistic_model)
print("Log of Odds (Coefficients):")
print(log_odds)

odds <- exp(log_odds)
print("Odds:")
print(odds)

# Decision Tree Model
decisiontree_model <- train(Revenue ~ ., 
                            data = shoppers_data, 
                            method = "rpart")

# Visualize the Decision Tree
rpart.plot(decisiontree_model$finalModel)

# KNN Model
knn_model <- train(Revenue ~ ., 
                   data = shoppers_data, 
                   method = "knn", 
                   preProcess = c("center", "scale"), 
                   trControl = trainControl(method = "cv", number = 10))

print(knn_model)

# Prediction using Logistic Regression Model
logistic_predictions <- predict(logistic_model, shoppers_data, type = "response")
# Convert the predicted probabilities to binary predictions (0 or 1)
logistic_predictions_binary <- ifelse(logistic_predictions > 0.5, 1, 0)

# Convert predictions to factors with the same levels as the 'Revenue' column
logistic_predictions_factor <- factor(logistic_predictions_binary, levels = levels(shoppers_data$Revenue))

# Create the confusion matrix
conf_matrix_logistic <- confusionMatrix(logistic_predictions_factor, shoppers_data$Revenue)

# Print confusion matrix for Logistic Regression
print("Confusion Matrix for Logistic Regression:")
print(conf_matrix_logistic)



# Prediction and Confusion Matrix for Decision Tree
# Make predictions using the decision tree model
dt_predictions <- predict(decisiontree_model, shoppers_data)

# Convert predictions to factors with the same levels as the 'Revenue' column
dt_predictions_factor <- factor(dt_predictions, levels = levels(shoppers_data$Revenue))

# Create the confusion matrix
conf_matrix_dt <- confusionMatrix(dt_predictions_factor, shoppers_data$Revenue)

# Print confusion matrix for Decision Tree
print("Confusion Matrix for Decision Tree:")
print(conf_matrix_dt)



# Prediction and Confusion Matrix for KNN
knn_predictions <- predict(knn_model, shoppers_data)
knn_predictions_factor <- factor(knn_predictions, levels = levels(shoppers_data$Revenue))
conf_matrix_knn <- confusionMatrix(knn_predictions_factor, shoppers_data$Revenue)
print("Confusion Matrix for KNN:")
print(conf_matrix_knn)


