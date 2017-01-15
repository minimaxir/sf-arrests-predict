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

lb <- as.numeric(iris$Species) - 1

set.seed(123)
trainIndex_i <- createDataPartition(iris$Species, p = 0.7,  list = FALSE,  times = 1)

train_i <- lgb.Dataset(data = data.matrix(iris[trainIndex_i, -5]), label = lb[trainIndex_i])
test_i <- lgb.Dataset.create.valid(train_i, data.matrix(iris[-trainIndex_i, -5]), label = lb[-trainIndex_i])

params <- list(objective = "multiclass", metric="multi_error")
valids <- list(test=test_i)

bst <- lgb.train(params, train_i, 20, valids,
                num_leaves = 4, learning_rate = 0.1, min_data=20, min_hess=20, num_class=3, verbose=1)

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


