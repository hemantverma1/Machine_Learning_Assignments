function [A, C] = votedPerceptron(Y, label, a, numEpoc) 	% Y is the matrix of augmented feature vectors, label defines label of each feature vector, 
								% a is the initial weight vector, numEpoc - number of times algo will run on the training data 
    c = 1;				% c saves the number of feature vectors this weight vector has survived
    A = [];                             % matrix of all weight vectors generated in the algo
    C = [];                             % matrix of lifetime (number of feature vectors this weight vector has survived) of every weight vector
    %disp(length(Y)), disp("hello"), disp(length(label));
    for i = 1:numEpoc                        % run till number of epochs given
        it = 1;
        for y = Y
            if((a'*y)*label(it) <= 0)
                A = [A, a];
                C = [C, c];
                a = a + y*label(it);
                c = 1;
            else
                c = c+1;
            end
            it = it+1;
        end
    end
    A = [A, a];
    C = [C, c];
end


