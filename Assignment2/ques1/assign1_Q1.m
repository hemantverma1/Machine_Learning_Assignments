data = [ [0.1;1.1;1] [6.8;7.1;1] [-3.5;-4.1;1] [2.0;2.7;1] [4.1;2.8;1] [3.1;5.0;1] [-0.8;-1.3;1] [0.9;1.2;1] [5.0;6.4;1] [3.9;4.0;1] [7.1;4.2;1] [-1.4;-4.3;1] [4.5;0.0;1] [6.3;1.6;1] [4.2;1.9;1] [1.4;-3.2;1] [2.4;-4.0;1] [2.5;-6.1;1] [8.4;3.7;1] [4.1;-2.2;1] ];
label = [1 1 1 1 1 1 1 1 1 1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1];

a = [0; 0; 0];
eita = 0.01;

count = 0;
iter = 0;

do
    true = 0;
    it = 1;
    iter = iter+1;
    for i = data
        if((a'*i)*label(it) <= 0)
            a = a + eita*i*label(it);
            true = 1;
            count = count+1;
        endif
        it = it+1;
    endfor
until(true == 0);

disp("Number of iterations");
disp(count), disp(iter);

hold on;

data1 = [ [0.1;1.1] [6.8;7.1] [-3.5;-4.1] [2.0;2.7] [4.1;2.8] [3.1;5.0] [-0.8;-1.3] [0.9;1.2] [5.0;6.4] [3.9;4.0] ];

data2 = [ [7.1;4.2] [-1.4;-4.3] [4.5;0.0] [6.3;1.6] [4.2;1.9] [1.4;-3.2] [2.4;-4.0] [2.5;-6.1] [8.4;3.7] [4.1;-2.2] ];

title ("Preceptron model for Question 1 a");

for i = data1
i = i';
plot(i(1), i(2));
endfor

%plot(data1);

for i = data2
i = i';
plot(i(1), i(2), 'ro');
endfor

%xlabel('Exam 1 score');
%ylabel('Exam 2 score');

%legend('Admitted', 'Not admitted');

A = a(1, 1);
B = a(2, 1);
C = a(3, 1);
x = -10:10;
plot(x, (-C-A*x)/B);
