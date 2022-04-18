rm(list = ls())
library(xgboost)
library(drat)
library(Matrix)
library(DiagrammeR)
data <- read.csv("data.csv")
train <- subset(data, !is.na(data$shot_made_flag))
test <- subset(data, is.na(data$shot_made_flag))
test.id <- test$shot_id
train$shot_id <- NULL
test$shot_id <- NULL
train.y <- train$shot_made_flag
train$shot_made_flag <- NULL
test$shot_made_flag <- NULL
pred <- rep(0, nrow(test))
trainM <- data.matrix(train, rownames.force = NA)
dtrain <- xgb.DMatrix(data = trainM, label = train.y, missing = NaN)
watchlist <- list(trainM=dtrain)

# KAGGLE MODEL
set.seed(1984)
param <- list(objectve  = "binary:logistic",
              booster = "gbtree",
              eval_metric = "logloss",
              eta = 0.035,
              max_depth = 4,
              subsample = 0.4,
              colsample_bytree = .4)
clf <- xgb.cv(params = param,
              data = dtrain,
              nrounds = 1500,
              verbose = 1, 
              watchlist = watchlist,
              maximize = FALSE,
              nfold = 3,
              early_stopping_rounds = 10,
              print_every_n = 1)
bestRound <- which.min(clf$evaluation_log$test_logloss_mean)
clf <- xgb.train(params = param, 
                 data = dtrain,
                 nrounds = bestRound,
                 verbose = 1, 
                 watchlist = watchlist,
                 maximize = FALSE)
log_loss <- min(clf$evaluation_log$trainM_logloss) #.593047

# OUR FINAL MODEL 
set.seed(1984)
param <- list(objectve  = "binary:logistic",
              booster = "dart",
              eval_metric = "logloss",
              eta = .035,
              max_depth = 4,
              subsample = .455,
              colsample_bytree = .385, gamma = .095)
clf <- xgb.cv(params = param,
              data = dtrain,
              nrounds = 1500,
              verbose = 1,
              watchlist = watchlist,
              maximize = FALSE,
              nfold = 3,
              early_stopping_rounds = 10,
              print_every_n = 1) 
bestRound <- which.min(clf$evaluation_log$test_logloss_mean)
clf <- xgb.train(params = param, 
                 data = dtrain,
                 nrounds = bestRound,
                 verbose = 1, 
                 watchlist = watchlist,
                 maximize = FALSE)
log_loss <- min(clf$evaluation_log$trainM_logloss) #.587114
importance_matrix <- xgb.importance(model = clf)
print(importance_matrix)
xgb.plot.importance(importance_matrix = importance_matrix)
testM <- data.matrix(test, rownames.force = NA)
preds <- predict(clf, testM)
submission <- data.frame(shot_id = test.id, shot_made_flag=preds)
# Kobe bryant career shooting percentage is .447
submission$shot_made_flag <- ifelse(submission$shot_made_flag>=.447, 1, 0)
write.csv(submission, "Kobe_Final_Submission.csv", row.names = F)

# AUC 
set.seed(1984)
param <- list(objectve  = "binary:logistic",
              booster = "dart",
              eval_metric = "auc",
              eta = .035,
              max_depth = 4,
              subsample = .455,
              colsample_bytree = .385, gamma = .095)
clf <- xgb.cv(params = param,
              data = dtrain,
              nrounds = 1500,
              verbose = 1, 
              rate_drop = .5,
              watchlist = watchlist,
              maximize = FALSE,
              nfold = 3,
              early_stopping_rounds = 10,
              print_every_n = 1,
              prediction = TRUE)
bestRound <- which.min(clf$evaluation_log$test_auc_mean)
thresh <- min(clf$evaluation_log$test_auc_mean) #.6325
it = which.max(clf$evaluation_log$test_auc_mean)
best.iter = clf$evaluation_log$iter[it]
library(pROC)
plot(pROC::roc(response = train.y,
               predictor = clf$pred,
               levels=c(0, 1)), lwd=1.5) 