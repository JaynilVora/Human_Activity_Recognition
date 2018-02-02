# Human Activity Recognition

library(caret)
library(randomForest)
library(rpart)
library(rpart.plot)

train1 <- read.csv('/Users/jaynilvora/Downloads/pml-training.csv', header=TRUE, sep=',', na.strings=c("NA","#DIV/0!",""))
test1 <- read.csv('/Users/jaynilvora/Downloads/pml-testing.csv', header=TRUE, sep=',', na.strings=c("NA","#DIV/0!",""))
set.seed(69)

# Exploratory Data Analysis
dim(train1)
dim(test1)
table(train1$classe)

# Data Cleaning
training <- train1[, -c(1:7)]
testing <- test1[, -c(1:7)]
dim(training);dim(testing)

# Convert everything except "classe" (last column) to numbers
features <- dim(training)[2]
suppressWarnings(training[,-c(features)] <- sapply(training[,-c(features)], as.numeric))
suppressWarnings(testing[,-c(features)] <- sapply(testing[,-c(features)], as.numeric))

# I have 2 options to deal with the NA values.
# 1st choice : I chose the option to delete any columns containing NAs. It is the simplest method but may lose information
training <- training[, colSums(is.na(training)) == 0]
testing <- testing[, colSums(is.na(testing)) == 0]
dim(training);dim(testing)

# 2nd choice: just remove columns with all or an excessive ratio of NAs. The threshold can be defined.
# We choose 80% threshold here
Threshold_NARatio = 0.8
ExcessiveNAsCol <- (colSums(is.na(training)) > (nrow(training) * Threshold_NARatio))
training <- training[!ExcessiveNAsCol]
testing <- testing[!ExcessiveNAsCol]
dim(training);dim(testing)

# I chose 2nd choice, as it is the only choice giving me same number of columns for both training & testing.

# Cross validation using a 75:25 partition
subsamples <- createDataPartition(y=training$classe, p=0.75, list=FALSE)
subTraining <- training[subsamples, ] 
subTesting <- training[-subsamples, ]

dim(subTraining)
dim(subTesting)

plot(subTraining$classe, col="green", main="Bar Plot of levels of the variable classes within the subTraining data set", xlab="classes levels", ylab="Frequency")

sample <- sample(subsamples,size=1000)
tr <- train2[sample,]
ts <- train2[-sample,]
dim(tr)
dim(ts)

# Creating the training model 

# I'm creating four training models with Naive Bayes, Logistic Regression, GBM & Random Forests and comparing the accuracy of predictions of each model
NaiveBayes <- train(classe~.,data=tr,method="nb")
LR <- glm(classe~ .,family=binomial("logit"), data=tr)
GBM <- train(classe~.,data=tr,method="gbm",verbose = FALSE)
RandomForest <- randomForest(classe ~. , data=tr, method="class")

#Naive Bayes Modeling
NBpred <- predict(NaiveBayes,newdata=ts)
NBaccuracy <- sum(NBpred == ts$classe)/length(ts$classe)
cat("NBaccuracy: ", NBaccuracy)
#Accuracy=56.05%

#Logistic Regression Modeling
LRpred <- predict(LR,ts, type='response')
LRaccuracy <- sum(LRpred == ts$classe)/length(ts$classe)
cat("LRaccuracy: ", LRaccuracy)
#Accuracy= error in execution(can't figure out why data has more levels than ref)

#GBM
GBMpred <- predict(GBM,newdata=ts)
GBMaccuracy <- sum(GBMpred == ts$classe)/length(ts$classe)
cat("GBMaccuracy: ", GBMaccuracy)
#Accuracy=83.69%

#Random Forest Modeling
RFpred <- predict(RandomForest, ts, type = "class")
confusionMatrix(RFpred, ts$classe)
#Accuracy=86.19%

#I tested the data on the original size and a smaller size, kyunki for different algorithms, we sometimes get higher accuracies, if we vary size of rows.
# From above tests, highest accuracy is for Random Forests.
# Hence, I used that for the final prediction, to decide which class do the 20 testing variables belong to.

predictfinal <- predict(RFpred, test2, type="class")
predictfinal