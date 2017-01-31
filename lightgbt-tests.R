# https://github.com/Microsoft/LightGBM/issues/214

library(lightgbm)

lb <- as.numeric(iris$Species) - 1

bst <- lightgbm(data = as.matrix(iris[, -5]), label = lb,
                num_leaves = 4, learning_rate = 0.1, nrounds = 20, min_data=20, min_hess=20,
                objective = "multiclass", metric="multi_error", num_class=3, verbose=0)

length(unlist(bst$record_evals))

bst <- lightgbm(data = as.matrix(iris[, -5]), label = lb,
                num_leaves = 4, learning_rate = 0.1, nrounds = 20, min_data=20, min_hess=20,
                objective = "multiclass", metric="multi_error", num_class=3, verbose=1)

length(unlist(bst$record_evals))

# categorical train/test

library(lightgbm)
library(caret)
library(jsonlite)

lb <- as.numeric(iris$Species) - 1

set.seed(123)
trainIndex <- createDataPartition(iris$Species, p = 0.5,  list = FALSE,  times = 1)

# w/o categorical

train_a <- lgb.Dataset(data = data.matrix(iris[trainIndex, -5]), label = lb[trainIndex])
test_a <- lgb.Dataset.create.valid(train_a, data.matrix(iris[-trainIndex, -5]), label = lb[-trainIndex])

params <- list(objective = "multiclass", metric="multi_error")
valids <- list(test=test_a)

bst <- lgb.train(params, train_a, 20, valids,
                num_leaves = 4, learning_rate = 0.5, min_data=5, min_hess=5, num_class=3, verbose=1)

toJSON(fromJSON(lgb.dump(bst)), pretty=T)

# w/ categorical

train_b <- lgb.Dataset(data = data.matrix(iris[trainIndex, -5]), label = lb[trainIndex])
test_b <- lgb.Dataset.create.valid(train_b, data.matrix(iris[-trainIndex, -5]), label = lb[-trainIndex])

params <- list(objective = "multiclass", metric="multi_error")
valids <- list(test=test_b)

bst <- lgb.train(params, train_b, 20, valids,
                 num_leaves = 4, learning_rate = 0.5, num_class=3, min_data=5, min_hess=5, verbose=1, categorical_feature = c(0:3))

toJSON(fromJSON(lgb.dump(bst)), pretty=T)


# test from docs

library(lightgbm)

data(agaricus.train, package='lightgbm')
train <- agaricus.train
dtrain <- lgb.Dataset(train$data, label=train$label)
data(agaricus.test, package='lightgbm')
test <- agaricus.test
dtest <- lgb.Dataset.create.valid(dtrain, test$data, label=test$label)
params <- list(objective="regression", metric="l2")
valids <- list(test=dtest)
model <- lgb.train(params, dtrain, 100, valids, min_data=1, learning_rate=1, early_stopping_rounds=10)


