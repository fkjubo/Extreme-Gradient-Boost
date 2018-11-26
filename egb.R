# importing library

library(caret)

# importing data 
# it's a kaggle competition
# they have two data frame train and test
# our test data doesn't have the target value which is NSP

train <- read.csv("train.csv", T, ",")
test <- read.csv("test.csv", T, ",")

# changing the target variable to factor variable

train$NSP <- as.factor(train$NSP)

# cheacking the structure of the train data

str(train)

# creating the model
# we are doing the model with cross validation

tuneConrol <- trainControl(method= "repeatedcv",
                          number= 10,
                          repeats = 5,
                          search = "grid")
             
# we also have a grid search on our model
# its basically tuning the model parameter

tune.grid <- expand.grid(eta= c(0.05,0.075,0.1),
                         nrounds= c(50,75,100),
                         max_depth= 6:8,
                         min_child_weight= c(2.0,2.5,2.75),
                         colsample_bytree= c(.3,.4,.5),
                         gamma= 0,
                         subsample = 1)

# we will do the model parallel for faster computation
# I have 2 cores so I will open two session

library(doParallel)

cl <- makePSOCKcluster(2)
registerDoParallel(cl)

# our model

model <- train(NSP ~ .,
               data= train,
               method= "xgbTree",
               trControl= tuneConrol,
               tuneGrid= tune.grid)

# closing the parallel session

stopCluster(cl)

# we don't have the target variable on test
# so first we will cheack the train data

predictTrain <- predict(model, train)

confusionMatrix(predictTrain, train$NSP)

# accuracy nearly 100%. Seems like over-fitting!

# making a csv file to submit on kaggle

predictionTest <- predict(model, test)
NSP <- predictionTest
output <- data.frame(NSP)

write.csv(output, "kaggle.csv", row.names = T)

