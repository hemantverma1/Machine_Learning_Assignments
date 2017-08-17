data = csvread("ionosphere.csv");	%reads whole file into data matrix
X = [];
label = [];

for i = 1:rows(data)						% i goes from 1 to number of rows in data
    feature = data(i, :);						% select ith row and all its columns
    if(feature(end) == 'b')
        label = [label, -1];						% append -1 to end of label matrix
    else
        label = [label, 1];
    endif
    X = [X, feature(:, 1:end-1)'];
endfor

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
        a = [zeros(1, rows(Y))];
	a = a';
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
            endif
        endfor
        
        [A, C] = votedPerceptron(trainingData, trainingDataLabel, a, epoc);
        a = [zeros(1, rows(Y))];
        a = a';
        a = perceptron(trainingData, trainingDataLabel, a, eita, epoc);
        %disp(a), disp("end");
        accVoted = accVoted + accuracyVotedPerceptron(A, C, testData, testDataLabel);
        accPercept = accPercept + accuracyPerceptron(a, testData, testDataLabel);
        %disp(accVoted), disp(accPercept);
    endfor
    accVoted = accVoted/k;
    accPercept = accPercept/k;
    disp(epoc), disp(accVoted), disp(accPercept), disp("");
    xlabel ("NumEpoc");
    ylabel ("Accuracy");
    accuracyVoted = [accuracyVoted, accVoted];
    accuracyVanilla = [accuracyVanilla, accPercept];
    %plot(epoc, accVoted, "b*");
    %plot(epoc, accPercept, "r.");
endfor
plot(epochs, accuracyVoted, "b--");
plot(epochs, accuracyVanilla, "r-");    
legend('Voted Perceptron', 'Vanilla Perceptron');
