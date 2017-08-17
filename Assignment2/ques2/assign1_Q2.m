data = csvread('breast-cancer-wisconsin.csv');	%reads whole file into data matrix
X = [];
label = [];

for i = 1:1:length(data)						% i goes from 1 to number of rows in data
    feature = data(i, :);						% select ith row and all its columns
    if(feature(end) == 2)
        label = [label, -1];						% append -1 to end of label matrix
    else
        label = [label, 1];
    end
    X = [X, feature(:, 2:end-1)'];
end

% Augmented feature vector
Y = vertcat(X, ones(1, length(X)));

%algo starts								
epochs = [10 15 20 25 30 35 40 45 50];
k = 10;

hold on;
accuracyVoted = [];
accuracyVanilla = [];
for i=1:length(epochs)              %run for every given epoch value
    epoc = epochs(i);

    %need to divide training set and testSet - k fold cross-validation
    CVO = cvpartition(label,'KFold',k);
    accVoted = 0.0;
    accPercept = 0.0;
    for j=1:k
        a = [0; 0; 0; 0; 0; 0; 0; 0; 0; 0];
        eita = 0.01;
        testData = [];
        testDataLabel = [];
        trainingData = [];
        trainingDataLabel = [];
        teIdx = test(CVO, j);
        teIdx = teIdx';
        for idx = 1:length(teIdx)
            if(teIdx(idx) == 1)  
                testData = [testData, Y(:, idx)];
                testDataLabel = [testDataLabel, label(idx)];
            else
                trainingData = [trainingData, Y(:, idx)];
                trainingDataLabel = [trainingDataLabel, label(idx)];
            end
        end
       
        [A, C] = votedPerceptron(trainingData, trainingDataLabel, a, epoc);
        a = [0; 0; 0; 0; 0; 0; 0; 0; 0; 0];
        c = perceptron(trainingData, trainingDataLabel, a, eita, epoc);
	accVotedTmp = accuracyVotedPerceptron(A, C, testData, testDataLabel);
	accPerceptTmp = accuracyPerceptron(c, testData, testDataLabel);    
        accVoted = accVoted + accVotedTmp;
        accPercept = accPercept + accPerceptTmp;
    end

    accVoted = accVoted/k;
    accPercept = accPercept/k;
    disp(epoc), disp(accVoted), disp(accPercept), disp('');
    xlabel ('NumEpoc');
    ylabel ('Accuracy');
    accuracyVoted = [accuracyVoted, accVoted];
    accuracyVanilla = [accuracyVanilla, accPercept];
    %plot(epoc, accVoted, 'b*');
    %plot(epoc, accPercept, 'r.');    
end   
plot(epochs, accuracyVoted, 'b--');
plot(epochs, accuracyVanilla, 'r-');
legend('Voted Perceptron', 'Vanilla Perceptron');


