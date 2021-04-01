library(tidyverse)
library(caret)
View(train)
ggplot(train)+geom_histogram(aes(x=Fare))
#Fares are roughly normally distributed. outliers: 20 obs above 300, 3 at 512.3292. 
ggplot(train)+geom_bar(aes(x=Cabin))
train$Cabin
#too many cabins. some people have mutiple cabins. perhaps they survived more as families
#would probably stick together?
multipleCabins <-  grepl(" ",train$Cabin, fixed = TRUE)#multi-cabin strings contain space
mean(train[multipleCabins,2])# 0.5833333
mean(train[-multipleCabins,2])# 0.3842697
mean(train[,2]) #0.3838384

#multicabin tickets had much higher chance of surviving, others had roughly same rate
#cabins have letters at the begginning. sorting by those:-
train1 <- mutate(train, Cabinletter=substr(train$Cabin,0,1))
mean(train1[train1$Cabinletter=="A",2]) #0.4666667
mean(train1[train1$Cabinletter=="B",2]) #0.7446809
mean(train1[train1$Cabinletter=="C",2]) #0.5932203
mean(train1[train1$Cabinletter=="D",2]) #0.7575758
mean(train1[train1$Cabinletter=="E",2]) #0.75
mean(train1[train1$Cabinletter=="F",2]) #0.6153846
mean(train1[train1$Cabinletter=="G",2]) #0.5
mean(train1[train1$Cabinletter=="T",2]) #0 
mean(train1[train1$Cabinletter!="",2])  #0.6666667
mean(train1[train1$Cabinletter=="",2])  #0.2998544

#passengers with tickets were more likely to survive in general. multiple cabins
#doesnt actually make a big difference. Perhaps they were prioritised as they were rich?
#checking if they were rich:-
ggplot(train1)+geom_histogram(aes(x=Fare, fill=Cabinletter))
#didnt help, cabin holders are too few. viewing it as proportion:
ggplot(train1)+geom_histogram(aes(x=Fare, fill=Cabinletter),position = "fill")
#cabin holders were definitely richer

#replace cabin variables with column that indicates whether or not the passenger has a 
#cabin
train2<- mutate(train1, Cabin1=(Cabin!=""))
train2$Cabin1 <- as.integer(train2$Cabin1)
train2<- select(train2, -Cabin,-Cabinletter)
train2<- select(train2, -PassengerId)
colnames(train2)[11]<-"Cabin"

#checking for class:
mean(train2[train2$Pclass==1,1]) #0.6296296
mean(train2[train2$Pclass==2,1]) #0.4728261
mean(train2[train2$Pclass==3,1]) #0.2423625
#higher class passengers survived more, in line with our previous observations

#tickets seem random, removing them
train2<-select(train2,-'Ticket')

#check siblings
mean(train2[train2$SibSp==0,1])#0.3453947
mean(train2[train2$SibSp==1,1])#0.5358852
mean(train2[train2$SibSp==2,1])#0.4642857
mean(train2[train2$SibSp==3,1])#0.25
mean(train2[train2$SibSp==4,1])#0.1666667
mean(train2[train2$SibSp==5,1])#0
mean(train2[train2$SibSp==6,1])#NaN
mean(train2[train2$SibSp==7,1])#NaN
mean(train2[train2$SibSp==8,1])#0
#2-3 siblings survived more, perhaps larger families couldn't fit in lifeboats and
# decided to stick together
train3<-mutate(train2, Survived=as.factor(Survived))
#train3 has survival as a factor, fill/color aesthetic couldn't be mapped to continuous
#variables for some reason
ggplot(train3,aes(x=SibSp,fill=Survived))+geom_bar(position="fill")
#we can see above, survival vs siblings forms a right skewed normal disitribution

#Also, females has a much higher chance of surviving:
ggplot(train3,aes(x=Sex,fill=Survived))+geom_bar(position="fill")
mean(train2[train2$Sex=="female",1])#0.7420382
mean(train2[train2$Sex=="male",1])  #0.1889081

#check if embarked has any effect of survival:
mean(train2$Survived[train2$Embarked=="S"])#0.3369565
mean(train2$Survived[train2$Embarked=="C"])#0.5535714
mean(train2$Survived[train2$Embarked=="Q"])#0.3896104
#C has a higher rate, check if sample size is reliable:
count(x = train2,Embarked=="C")
#168 observations, treating C and S as one group, Q as another
train2 <- mutate(train2, "EmbarkedQ"=as.integer(Embarked=="Q"))
train2 <- select(train2, -Embarked)

#177 NA values in age, replacing them with average
ggplot(train)+geom_density(aes(Age))
#distribution is skewed to the right,using median, not mean
train2$Age[is.na(train$Age)]<-median(train$Age, na.rm=TRUE)

#remove upper half of fares
train2 <- train2[-(train2$Fare>(max(train2$Fare)/2)),]
#remove SibSp>4
train2 <- train2[-(train2$SibSp>4),]

#normalising fare,age, etc
train2$Fare <- (train2$Fare-min(train2$Fare))/(max(train2$Fare)-min(train2$Fare))
train2$Age <- (train2$Age-min(train2$Age))/(max(train2$Age)-min(train2$Age))
train2$SibSp <- (train2$SibSp-min(train2$SibSp))/(max(train2$SibSp)-min(train2$SibSp))

#convert survival to factors, otherwise CARET will treat data as regression problem
#instead of classification, but train2 is preserved so mean() can be used.
train3 <- train2
train3$Survived <- as.factor(train3$Survived)
train3 <- select(train3, -`Name`)

#Change Sex from char to int
train3$Sex <- as.integer(train3$Sex=="male")

#Begin Machine Learning:
set.seed(4919)
flag <- createDataPartition(train3$Survived, p=0.75, list=FALSE)
training <- train3[flag,]

#remove upper half of Fares for more accuracy, they're mostly outliers
training<-training[training$Fare<0.5,]

testing  <- train3[-flag,]
# knn_model <- train(select(training,-`Survived`),training[,1],method='knn')
# prediction_knn <- predict(knn_model, testing[,-1])
#   #For Detailed Information Use:
# confusionMatrix(prediction_knn,testing[,1])
#using different model:
# svmLinear_model <- train(select(training,-`Survived`),training[,1],method='svmLinear')
# prediction_svmLinear <- predict(svmLinear_model, testing[,-1])
# confusionMatrix(prediction_svmLinear,testing[,1])

#rf somehow better than ranger for randomforest
ranger_model <- train(select(training,-`Survived`),training[,1],method='rf',preProc=c("center","scale"))
prediction_ranger <- predict(ranger_model, testing[,-1])
confusionMatrix(prediction_ranger,testing[,1])