function c = perceptron(Y, label, a, eita, numEpoc) 		% Y is the matrix of augmented feature vectors, label defines label of each feature vector, 
								% a is the initial weight vector, eita - learning rate, numEpoc - number of times algo will run on the training data 
    c = a;
    for i = 1:numEpoc                        % run till number of epochs given
        it = 1;
        for y = Y
            if((c'*y)*label(it) <= 0)
                c = c + eita*y*label(it);
            end
            it = it+1;
        end
    end
end


