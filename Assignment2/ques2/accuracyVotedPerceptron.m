function acc = accuracyVotedPerceptron(A, C, testData, testDataLabels) % A - matrix of all weight vectors generated in the algo, C - lifetime of every wt. vector
    acc = 0.00000000;
    P = predictionVotedPerceptron(A, C, testData);
    for i = 1:length(P)
        if(P(i)*testDataLabels(i) > 0)
            acc = acc+1.000;
        end
    end
    acc = acc/length(P);
end

function P = predictionVotedPerceptron(A, C, testData) % A - matrix of all weight vectors generated in the algo, C - lifetime of every wt. vector    
    P = [];                                                         % prediction matrix
    for y = testData
        prediction = 0;
        j = 1;
        for a = A
            if(a'*y < 0)
                sigN = -1;                                          % sigN is sign when I put the point in my hyperplane
            elseif(a'*y == 0)
                sigN = 0;
            else
                sigN = 1;
            end
            prediction = prediction + sigN*C(j);
            j = j+1;
        end    

        if(prediction < 0)
            prediction = -1;
        else
            prediction = 1;
        end
        P = [P, prediction];
    end
end


