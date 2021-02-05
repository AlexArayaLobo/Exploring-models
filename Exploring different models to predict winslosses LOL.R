#Exploring different models to predict wins/losses LOL

#Libraries
library(tidyverse)
library(recipes)
library(tidymodels)
library(dplyr)
library(tidyr)
library(broom)
library(DataExplorer)
library(corrplot)
library(ggplot2)
library(lmtest)
library(pscl)
library(caret)
library(naivebayes)
library(randomForest)
library(fastDummies)

#Get the data
DataLOL <- read.csv("high_diamond_ranked_10min.csv") 

#Select blue team features (exclude red team, it is enough with one team due to de characteristics of the game).
DataBlueTeam <-
  DataLOL %>% 
    select(blueWins:blueGoldPerMin)

#EDA
names(DataBlueTeam)
head(DataBlueTeam)
summary(DataBlueTeam)
str(DataBlueTeam)
glimpse(DataBlueTeam)
DataBlueTeam$blueWins<-as.factor(DataBlueTeam$blueWins) #bluewins as factor
DataBlueTeam$blueFirstBlood<-as.factor(DataBlueTeam$blueFirstBlood) #firstblood as factor
create_report(DataBlueTeam) #Automate EDA

#Correlation
CorrMatrix <-cor(DataBlueTeam %>% 
                   select(-blueWins, - blueFirstBlood))
corrplot(CorrMatrix,method = "color",type = "full",
         order = "hclust",addCoef.col = "black",
         tl.srt = 40,tl.cex = 0.7,number.cex = 0.55)

#Data Preprocessing

#Drop unnecessary and correlated columns
ncol(DataBlueTeam)

DataBlueTeam <- 
  DataBlueTeam %>% 
  dummy_cols() %>%
  select(-blueFirstBlood,-10,-17,-18,-19,-20, -21, -22)  
  
#Scale and center. This could be made by a recipe (tidymodels). But this time we are not going to apply tidymodels.
attach(DataBlueTeam)
DataBlueTeam$blueWardsDestroyed=scale(blueWardsDestroyed, center = TRUE, scale = TRUE)
DataBlueTeam$blueKills=scale(blueKills, center = TRUE, scale = TRUE)
DataBlueTeam$blueAssists=scale(blueAssists, center = TRUE, scale = TRUE)
DataBlueTeam$blueTowersDestroyed=scale(blueTowersDestroyed, center = TRUE, scale = TRUE)
DataBlueTeam$blueAvgLevel=scale(blueAvgLevel, center = TRUE, scale = TRUE)
DataBlueTeam$blueTotalMinionsKilled=scale(blueTotalMinionsKilled, center = TRUE, scale = TRUE)
DataBlueTeam$blueWardsPlaced=scale(blueWardsPlaced, center = TRUE, scale = TRUE)
DataBlueTeam$blueDeaths=scale(blueDeaths, center = TRUE, scale = TRUE)
DataBlueTeam$blueEliteMonsters=scale(blueEliteMonsters, center = TRUE, scale = TRUE)
DataBlueTeam$blueTotalGold=scale(blueTotalGold, center = TRUE, scale = TRUE)
DataBlueTeam$blueTotalExperience=scale(blueTotalExperience, center = TRUE, scale = TRUE)
DataBlueTeam$blueTotalJungleMinionsKilled=scale(blueTotalJungleMinionsKilled, center = TRUE, scale = TRUE)
DataBlueTeam$blueDragons=scale(blueDragons, center = TRUE, scale = TRUE)

#Spliting Data
set.seed(123)
Data_split <- initial_split(DataBlueTeam, strata = blueWins)
Data_train <- training(Data_split)
Data_test <- testing(Data_split)

#Balanced Factors
Data_train %>%
  count(blueWins)

#Fit Logistic regression
glm1 <- glm(blueWins ~.,data = Data_train, family = 'binomial')
summary(glm1) 

"
#Try AIC
step(glm1)

#New model usign AIC criteria
mymodel2<-glm(blueWins ~ blueWardsPlaced + blueDeaths + blueAssists + 
                blueEliteMonsters + blueTowersDestroyed + blueTotalGold + 
                blueTotalExperience + blueTotalMinionsKilled,data = Data_train, family = 'binomial')
summary(mymodel2)

#Likelihood Ratio Test
lrtest(glm1,mymodel2)
#New model did not improved compared to the old model. Keep "mymodel1"
"

#Making predictions
Data_train$pred <- predict(glm1, newdata=Data_train, type="response")
Data_test$pred <- predict(glm1, newdata=Data_test, type="response")

#Choose threshold
ggplot(Data_train, aes(x=pred, color=blueWins, linetype=blueWins)) +
  geom_density() #0.5 aprox

#Confusion Matrix
ctab.test <- table(pred=Data_test$pred>0.5, blueWins=Data_test$blueWins) 
ctab.test 

#Accuracy
pred.classes<-ifelse(Data_test$pred>0.5,"1","0")
AccLog<-mean(pred.classes == Data_test$blueWins)
AccLog

#ODDS Ratio
tidy(glm1, exponentiate = TRUE, conf.level = 0.95)

#Pseudo R2: McFadden
pR2(glm1) #Low McFadden, low predictive power.

#Comparing other techniques
#Reload Data_train

#LDA
modlda = train(blueWins~.,data=Data_train,method="lda")
plda = predict(modlda,Data_test)
CMlda<-confusionMatrix(data=plda, Data_test$blueWins)

#Naive Bayes
ModelNaiveBayes<-naive_bayes(blueWins~.,data = Data_train)
predNB<-predict(ModelNaiveBayes,Data_test)
CMnaive<-confusionMatrix(data=predNB, Data_test$blueWins)

#RandomForest
modeloRF<-train(blueWins~.,data = Data_train,method="rf")
predRF <- predict(modeloRF,Data_test)
CMrf<-confusionMatrix(data=predRF, Data_test$blueWins)

#Boosting
modelosBoosting<-train(blueWins~.,data=Data_train, method="gbm",verbose=FALSE)
test_pred_boost<-predict(modelosBoosting,Data_test)
CMboost<-confusionMatrix(table(test_pred_boost, Data_test$blueWins))

#Support Vector Machine
svm_Linear <- train(blueWins ~.,data = Data_train, method="svmLinear")
test_pred_svm <- predict(svm_Linear,Data_test)
CMsvm<-confusionMatrix(table(test_pred_svm, Data_test$blueWins))

#Evaluate models accuracy
Evaluations<-list(CMlda$overall["Accuracy"],CMnaive$overall["Accuracy"],
                  CMsvm$overall["Accuracy"],CMboost$overall["Accuracy"],
                  CMrf$overall["Accuracy"])

names(Evaluations)<-c("LDA","Naive","SVM","Boost","RF")
Evaluations #SVM had the higher accuracy.

#Variable importance according to SVMmodel (higher accuracy)
varImp(svm_Linear) #Most important variables: blueTotalGold,blueTotalExperience,blueAvgLevel

#New SVM model
TrainControlSVM<-trainControl(method = "repeatedcv", 
                              number = 10, 
                              repeats = 3)

svm_Linear2<-train(blueWins~.,data=Data_train,
                   method="svmLinear",
                   trControl=TrainControlSVM,
                   tuneLength=10)
svm_Linear2
predSVM2<-predict(svm_Linear2,Data_test)
CMsvm2<-confusionMatrix(table(predSVM2, Data_test$blueWins))

grid <- expand.grid(C = c(0,0.01, 0.05, 0.1, 0.25, 0.5, 
                          0.75, 1, 1.25, 1.5, 1.75, 2,5))
svmLinear2_Grid <- train(blueWins ~., data = Data_train, 
                         method = "svmLinear",
                         trControl=TrainControlSVM,
                         tuneGrid = grid,
                         tuneLength = 10)
svmLinear2_Grid
plot(svmLinear2_Grid)
test_pred_SVM2Grid<-predict(svmLinear2_Grid,Data_test)
CMSVM2Grid<-confusionMatrix(table(test_pred_SVM2Grid, Data_test$blueWins))
#New Model did not improved.
