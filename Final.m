load('CrimeStats.mat');
figure(1);
scatter(CRIME,MEDV);
xlabel('Crime');
ylabel('Median Value');
figure(2);
scatter(BLACKS, MEDV);
xlabel('African American Population');
ylabel('Median Value');
figure(3);
scatter(BLACKS, CRIME);
xlabel('African American Population');
ylabel('Crime');
SlopeYIntercept = polyfit(CRIME, MEDV,1);
Slope = SlopeYIntercept(:,1);
YIntercept = SlopeYIntercept(:,2);

%Code written by Stephen Trac 
%Seperate training set and test set 
%Create an array of nonrepeating random values from 1 - 392
randomIndex = randperm(392,392); 
%Take the first ~70% of randomIndex
trainingIndex = randomIndex(1,1:274);
%Take the last ~30% of randomIndex
testingIndex = randomIndex(1,275:392);
%Seperate both data sets by a testing section and a training section for
%the models. 
HPTraining = horsepower(trainingIndex)'; 
HPTest = horsepower(testingIndex)'; 
mpgTraining = mpg(trainingIndex)'; 
mpgTest = mpg(testingIndex)'; 
%% Note everything past this point will imitate the reignItIn assignment for the sake of "finish on time"
%We will assume that the x values = horsepower and y values = mpg 
%% First order model selected y = w0 + w1x
    %Matrix form of data will be y = Aw
    %Create matrix A in order to find weights w using vector x
        A1 = [ones(length(HPTraining),1),HPTraining];
    %calculate w using y and matrix A 
        w1 = A1\mpgTraining;
    %Filling in the original model, y = 40.4918 - 0.1627x
    %Visualizing data using xTest, aTest and yTest 
    %xTest is created to input into formula in order to plot in matlab
    %note: range of xTest is based on the max/min values of horsepower =
    %230/46
        xTest = [-250:1:250];
    %aTest is created to find the yTest 
    %note: I believe this was left out in the lecture slides but it appears
    %that concatination only works if equal number of rows are present in two
    %matrices so xTest has to be transposed because it's a row vector 
        a1Test = [ones(length(xTest),1),xTest'];
    %Calculating yTest using the matrix aTest and our w; basically plugging
    %these values into our model 
        y1Test = a1Test*w1;
    %Calculating the SSE
        y1Prediction =40.4918 - 0.1627*HPTraining;
        y1Prediction1 =40.4918 - 0.1627*HPTest;
        errorTraining1 = sum((y1Prediction - mpgTraining).^2)
        fprintf('Sum squared error for 1st order linear model training is: %s',errorTraining1)
        errorTest1 = sum((y1Prediction1 - mpgTest).^2) 
        fprintf('Sum squared error for 1st order linear model testing is: %s',errorTest1)
%% Second Order Regression where y = w0 + w1x + w2x^2 
    %Matrix form of data will be y = Aw
    %Create matrix A in order to find weights w using vector x
        A2 = [ones(length(HPTraining),1),HPTraining,HPTraining.^2];
    %calculate w using y and matrix A 
        w2 = A2\mpgTraining;
    %Filling in the original model, y = 56.7412 - 0.4599x + 0.0012x^2
    %Visualizing data using xTest, aTest and yTest 
    %xTest is created to input into formula in order to plot in matlab
    %note: range of xTest is based on the max/min values of horsepower =
    %230/46
        xTest = [-250:1:250];
    %aTest is created to find the yTest 
    %note: I believe this was left out in the lecture slides but it appears
    %that concatination only works if equal number of rows are present in two
    %matrices so xTest has to be transposed because it's a row vector 
        a2Test = [ones(length(xTest),1),xTest',(xTest.^2)'];
    %Calculating yTest using the matrix aTest and our w; basically plugging
    %these values into our model 
        y2Test = a2Test*w2;
    %Calculating the SSE
        y2Prediction =56.7412 - 0.4599*HPTraining + 0.0012*(HPTraining.^2);
        y2Prediction2 =56.7412 - 0.4599*HPTest + 0.0012*(HPTest.^2);
        errorTraining2 = sum((y2Prediction - mpgTraining).^2)
        fprintf('Sum squared error for 2nd order linear model training is: %s',errorTraining2)
        errorTest2 = sum((y2Prediction2 - mpgTest).^2) 
        fprintf('Sum squared error for 2nd order linear model testing is: %s',errorTest2)
%% Third Order Model where y = w0 + w1x1 + w2x^2 + w3x^3
    %Matrix form of data will be y = Aw
    %Create matrix A in order to find weights w using vector x
        A3 = [ones(length(HPTraining),1),HPTraining,HPTraining.^2,HPTraining.^3];
    %calculate w using y and matrix A 
        w3 = A3\mpgTraining;
    %Filling in the original model, y=58.1705-0.4990x+0.0015x^2-8.3774e-07x^3
    %Visualizing data using xTest, aTest and yTest 
    %xTest is created to input into formula in order to plot in matlab
    %note: range of xTest is based on the max/min values of horsepower =
    %230/46
        xTest = [-250:1:250];
    %aTest is created to find the yTest 
    %note: I believe this was left out in the lecture slides but it appears
    %that concatination only works if equal number of rows are present in two
    %matrices so xTest has to be transposed because it's a row vector 
        a3Test = [ones(length(xTest),1),xTest',(xTest.^2)',(xTest.^3)'];
    %Calculating yTest using the matrix aTest and our w; basically plugging
    %these values into our model 
        y3Test = a3Test*w3;
    %Calculating the SSE
        y3Prediction =58.1705-0.4990*HPTraining+0.0015*(HPTraining.^2)-(8.3774e-07)*(HPTraining.^3);
        y3Prediction3 =58.1705-0.4990*HPTest+0.0015*(HPTest.^2)-(-8.3774e-07)*(HPTest.^3);
        errorTraining3 = sum((y3Prediction - mpgTraining).^2)
        fprintf('Sum squared error for 3rd order linear model training is: %s',errorTraining3)
        errorTest3 = sum((y3Prediction3 - mpgTest).^2) 
        fprintf('Sum squared error for 3rd order linear model testing is: %s',errorTest3)
        
%% Now we plot the three models with the data inputs 
figure;
%Plot First order model in top left hand graph
subplot(2,2,1);
hold on; 
plot(xTest,y1Test); 
%The data points will be in * while the test will be in o
plot(HPTest,mpgTest,'or');
plot(horsepower',mpg','c*');
hold off; 
%Plot second order model in top right hand graph
subplot(2,2,2); 
hold on; 
plot(xTest,y2Test);
%The data points will be in * while the test will be in o
plot(HPTest,mpgTest,'or');
plot(horsepower',mpg','c*');
hold off; 
%Plot third order model in bottom left hand graph 
subplot(2,2,3); 
hold on; 
plot(xTest,y3Test);
%The data points will be in * while the test will be in o
plot(HPTest,mpgTest,'or');
plot(horsepower',mpg','c*');
hold off; 
%Note SSE for all three are printed in Command Window