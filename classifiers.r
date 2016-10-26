#Required Packages
require(rpart)
require(rpart.plot)
require(neuralnet)
require(e1071)

#including package libraries
library(rpart)
library(rpart.plot)
library(neuralnet)
library(e1071)

#reading input
b <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", sep = ",", header = FALSE)
names(b) <- c("Sepal.Length","Sepal.Width","Petal.Length","Petal.Width","Species")
apply(b, MARGIN = 2, FUN = function(x) sum(is.na(x)))
maxs = apply(b[,-5], MARGIN = 2, max)
mins = apply(b[,-5], MARGIN = 2, min)
scaled = as.data.frame(scale(b[-5], center = mins, scale = maxs - mins))
input <- as.data.frame(cbind(scaled,b[,5]))
names(input) <- c("Sepal.Length","Sepal.Width","Petal.Length","Petal.Width","Species")
head(input)
randomdata <- sample(150,120)
traindata <- input[randomdata,]
testdata <- input[-randomdata,]

#Decision Tree Classifier
dt <- rpart(Species~.,traindata, method = "class")
printcp(dt)
rpart.plot(dt, type =4, extra = 102)
p<- predict(dt, testdata, type="class")
table(testdata[,5],p)
plot(testdata[,5],p)

#Perceptron Classifier
nnet_train <- traindata
# Binarize the categorical output
nnet_train <- cbind(nnet_train, traindata$Species == 'setosa')
nnet_train <- cbind(nnet_train, traindata$Species == 'versicolor')
nnet_train <- cbind(nnet_train, traindata$Species == 'virginica')
names(nnet_train)[6] <- 'setosa'
names(nnet_train)[7] <- 'versicolor'
names(nnet_train)[8] <- 'virginica'
head(nnet_train)
nn <- neuralnet(setosa+versicolor+virginica ~ Sepal.Length+Sepal.Width+Petal.Length+Petal.Width,data=nnet_train,hidden=0)
plot(nn)
pred <- compute(nn, testdata[-5])
pred.weights <- pred$net.result
neural <- apply(pred.weights,1,which.max)
prediction <- c('setosa', 'versicolor', 'virginica')[neural]
table(prediction, testdata$Species)

#SVM Classifier
x <- subset(traindata,select = -Species)
y <- traindata$Species
svmmodel <- svm(Species~.,data = traindata)
summary(svmmodel)
pred <- predict(svmmodel,x)
table(pred,y)
#Finding best cost and gamma values for tuning the data
svm_tune <- tune(svm, train.x=x, train.y=y, kernel="linear", ranges=list(cost=10^(-1:2), gamma=c(.5,1,2)))
print(svm_tune)
m <- subset(testdata,select = -Species)
n <- testdata$Species
svmmodel_after_tune <- svm(Species~.,data = testdata, kernel='linear', cost=5, gamma = 0.3)
summary(svmmodel_after_tune)
pred <- predict(svmmodel_after_tune,m)
table(pred,n)
plot(pred,n)

#Neural Net Classifier
nnet_train <- traindata
# Binarize the categorical output
nnet_train <- cbind(nnet_train, traindata$Species == 'setosa')
nnet_train <- cbind(nnet_train, traindata$Species == 'versicolor')
nnet_train <- cbind(nnet_train, traindata$Species == 'virginica')
names(nnet_train)[6] <- 'setosa'
names(nnet_train)[7] <- 'versicolor'
names(nnet_train)[8] <- 'virginica'
head(nnet_train)
nn <- neuralnet(setosa+versicolor+virginica ~ Sepal.Length+Sepal.Width+Petal.Length+Petal.Width,data=nnet_train,hidden=2, threshold = 0.1)
plot(nn)
pred <- compute(nn, testdata[-5])
pred.weights <- pred$net.result
neural <- apply(pred.weights,1,which.max)
prediction <- c('setosa', 'versicolor', 'virginica')[neural]
table(prediction, testdata$Species)

#Naive Bayes Classifier
naivemodel <- naiveBayes(traindata[,1:4],traindata[,5])
pred.class <- predict(naivemodel,testdata[,-5])
table(pred.class,testdata[,5],dnn = list('predicted','actual'))

naivemodel$apriori / sum(naivemodel$apriori)
plot(0:8, xlim=c(0.1,5), col="red", ylab="density",type="n", xlab="Petal Length",main="Petal length distribution for each species")
curve(dnorm(x, naivemodel$tables$Petal.Length[1,1], naivemodel$tables$Petal.Length[1,2]), add=TRUE, col="red")
curve(dnorm(x, naivemodel$tables$Petal.Length[2,1], naivemodel$tables$Petal.Length[2,2]), add=TRUE, col="blue")
curve(dnorm(x, naivemodel$tables$Petal.Length[3,1], naivemodel$tables$Petal.Length[3,2]), add=TRUE, col ="green")
legend("topright", c("setosa", "versicolor", "virginica"), col = c("red","blue","green"), lwd=1)

plot(0:8, xlim=c(0.1,5), col="red", ylab="density",type="n", xlab="Petal Width",main="Petal Width distribution for each species")
curve(dnorm(x, naivemodel$tables$Petal.Width[1,1], naivemodel$tables$Petal.Width[1,2]), add=TRUE, col="red")
curve(dnorm(x, naivemodel$tables$Petal.Width[2,1], naivemodel$tables$Petal.Width[2,2]), add=TRUE, col="blue")
curve(dnorm(x, naivemodel$tables$Petal.Width[3,1], naivemodel$tables$Petal.Width[3,2]), add=TRUE, col ="green")
legend("topright", c("setosa", "versicolor", "virginica"), col = c("red","blue","green"), lwd=1)

plot(0:4, xlim=c(0.1,5), col="red", ylab="density",type="n", xlab="Sepal Length",main="Sepal length distribution for each species")
curve(dnorm(x, naivemodel$tables$Sepal.Length[1,1], naivemodel$tables$Sepal.Length[1,2]), add=TRUE, col="red")
curve(dnorm(x, naivemodel$tables$Sepal.Length[2,1], naivemodel$tables$Sepal.Length[2,2]), add=TRUE, col="blue")
curve(dnorm(x, naivemodel$tables$Sepal.Length[3,1], naivemodel$tables$Sepal.Length[3,2]), add=TRUE, col ="green")
legend("topright", c("setosa", "versicolor", "virginica"), col = c("red","blue","green"), lwd=1)

plot(0:4, xlim=c(0.1,5), col="red", ylab="density",type="n", xlab="Sepal Width",main="Sepal Width distribution for each species")
curve(dnorm(x, naivemodel$tables$Sepal.Width[1,1], naivemodel$tables$Sepal.Width[1,2]), add=TRUE, col="red")
curve(dnorm(x, naivemodel$tables$Sepal.Width[2,1], naivemodel$tables$Sepal.Width[2,2]), add=TRUE, col="blue")
curve(dnorm(x, naivemodel$tables$Sepal.Width[3,1], naivemodel$tables$Sepal.Width[3,2]), add=TRUE, col ="green")
legend("topright", c("setosa", "versicolor", "virginica"), col = c("red","blue","green"), lwd=1)
