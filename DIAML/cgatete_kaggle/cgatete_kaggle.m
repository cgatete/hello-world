% Kaggle compet
%loaded data into matlab
titanicKaggle = readtable('test.csv');
titanicTrain = readtable('train.csv');

%data for training
PassengerClass = titanicTrain.Pclass;
SexData = string(titanicTrain.Sex);
fare = titanicTrain.Fare;
sib = titanicTrain.SibSp;
Survived = titanicTrain.Survived;

AgeData = titanicTrain.Age;
sexDataDouble = double(strncmp(SexData, 'female', 3));
predictors = [PassengerClass,AgeData,  sexDataDouble, fare, sib];
PassengerID = titanicTrain.PassengerId;

%data for testing 
PassengerClassTest = titanicKaggle.Pclass;
SexDataTest = string(titanicKaggle.Sex);

AgeDataTest = titanicKaggle.Age;
fareTest = titanicKaggle.Fare;
sibTest = titanicKaggle.SibSp;

sexDataDoubleTest = double(strncmp(SexDataTest, 'female', 3));
predictorsTest = [PassengerClassTest, AgeDataTest,sexDataDoubleTest, fareTest, sibTest ];
PassengerIDTest = titanicKaggle.PassengerId;

%Constructing the model using classification tree
tree = fitctree(predictors, Survived);

% Finding the optimal pruning level
[~,~,~,bestlevel] = cvLoss(tree,'subtrees','all','treesize','min');
pctree = prune(tree,'Level',bestlevel);

[Survived1,Sfit] = predict(pctree, predictorsTest)

kaggleh = [PassengerIDTest, Survived1]
csvwrite('KaggleSub9.csv',kaggleh)



