C1 = [[3;3] [3;0] [2;1] [0;2]];
C2 = [[-1;1] [0;0] [-1;-1] [1;0]];
%disp(length(C1)); %<--- length of dataset
%disp(length(C2)); %<---- length of dataset
UC1=[]; %<--- mean-element of class1
UC2=[]; %<--- mean-element of clss 2
%u1 = sum(C1,2);
%u2 = sum(C2,2);
%u1 = u1/length(C1); %<--- mean of first class
%u2 = u2/length(C2);
u1 = mean(C1,2);
u2 = mean(C2,2);
for c1=C1
d = u1-c1;
UC1 = [UC1,d];
end
for c2=C2
d = u2-c2;
UC2 = [UC2,d];
end
S1 =zeros(2,2); %<---- sum scatter matrix of class1
S2 =zeros(2,2);
for U = UC1
A = U*U';
S1 = S1+A;
end
for U = UC2
A = U*U';
S2 = S2+A;
end

SW = S1+S2; %<----SCATTER MATRIX
U = u1-u2;
ISW = inv(SW);
e = ISW*U;
%disp(SW);
%disp(ISW);
%disp(e);
%disp(norm(e));
hold on;
e = e ./ norm(e);
plot([-2 e(1)*5] , [-2 e(2)*5]);

for i = C1
plot(i(1),i(2),"ro");
doT = dot(i, e);
projection = doT*e;
plot(projection(1), projection(2), "c");
end

for i = C2
plot(i(1),i(2),"g^");
end
%disp(u1);
%disp(u2);
%disp(UC1);
%disp(UC2);
