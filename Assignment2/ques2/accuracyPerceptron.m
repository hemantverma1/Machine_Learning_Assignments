function acc = accuracyPerceptron(a, testData, label)
    acc = 0.00000000;
    it = 1;
    for y = testData
        if((a'*y)*label(it) > 0)
            acc = acc+1.000;
        end
        it = it+1;
    end
    acc = acc/length(testData);
end
