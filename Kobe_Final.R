## Data Cleaning ##
rm(list=ls())
options(scipen=99)

data <- read.csv("data.csv")

# Drop irrelevant columns
data <- subset(data, select = -c(opponent, action_type, game_event_id, lat, lon, shot_zone_range, team_name, team_id, game_date)) 
#drop action_type: includes combined shot type 
#drop game_event_id
#drop lat, lon
#drop shot_zone_range because binned version of shot distance
#drop team_id and team_name
#drop game_date (information on order of shots from game in shot_id)

# Updated dimension of the dataset
dim(data) #30697    16

# View data
str(data)

# Create dummy variables for categorical variables
library(fastDummies)
data$playoffs <- as.character(data$playoffs)
data$period <- as.character(data$period)
data <- dummy_cols(data, select_columns = c("combined_shot_type","playoffs","period","shot_type"))
data <- subset(data, select = -c(combined_shot_type, playoffs, period, shot_type))

# Combine units: minutes remaining into seconds remaining and drop
data$seconds_remaining <- 60 * data$minutes_remaining + data$seconds_remaining

# Turn seconds remaining into a dummy variable where 1 is <5 seconds
data$last_5_sec_in_period <- data$seconds_remaining < 5
data$last_5_sec_in_period = factor(data$last_5_sec_in_period,
                                   levels = c("FALSE", "TRUE"),
                                   labels = c(0, 1))
data$last_5_sec_in_period <- as.character(data$last_5_sec_in_period)   
data <- subset(data, select = -c(minutes_remaining))

# Convert season to numeric
data$season <- as.numeric(sapply(strsplit(as.character(data$season), "-"), "[[", 1))

# Use matchup to determine home vs away (dummy code) and drop
data$home <- as.numeric(grepl("vs.", data$matchup, fixed = TRUE))
data$home <- as.factor(data$home)
data <- subset(data, select = -c(matchup))

# Dummy code last_5_sec_in_period and home
data <- dummy_cols(data, select_columns = c("last_5_sec_in_period","home"))
data <- subset(data, select = -c(last_5_sec_in_period, home))

#opponent (character)- leave in for now for visualizing, not for modeling (doesn't effect)
#maybe drop shot_zone_basic and shot_zone_area: lat and lon already have this information (leave for now)- choose one to keep?
#drop opponent, shot_zone_basic and shot_zone_area, and shot_id in models

# Rename column names that have spaces in them
library(dplyr)
data <- data %>% 
  rename(
    combined_shot_type_Bank_Shot = "combined_shot_type_Bank Shot",
    combined_shot_type_Hook_Shot = "combined_shot_type_Hook Shot",
    combined_shot_type_Jump_Shot = "combined_shot_type_Jump Shot",
    combined_shot_type_Tip_Shot = "combined_shot_type_Tip Shot",
    shot_type_2PT_Field_Goal = "shot_type_2PT Field Goal",
    shot_type_3PT_Field_Goal = "shot_type_3PT Field Goal"
    
  )

# Insignificant periods
data <- subset(data, select = -c(period_5, period_6, period_7))

########################################### FINAL MODEL ############################
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