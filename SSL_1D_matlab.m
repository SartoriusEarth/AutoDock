clear all;
close all;

p=0.5;
sz = 1;
tick = sz/1000;
x = 0:tick:sz;
y = floor(2*((log(x)/log(p))-(floor(log(x)/log(p))))); % SSL square wave

figure;
plot(x,y);
axis([0,1,-0.5,1.5]);

% create input array
w=500;
input = zeros(1,w);
for i = 1:99
    input(i+100) = sp(i/100,p);
end
input(300:400) = 1;
figure;
plot(input);
axis([0,w,-0.5,1.5]);

% % maximize p p^2 p^3...
% % maximize mismatch p^1/2 p^3/2 p^5/2...
result = zeros(1,w-1);
ri = 1; % result index
for d = 1:w-100
    sumMatch = 0;
    for i = 1:50
         sumMatch = sumMatch + abs(input(d+i) - input(floor(d+i*p)));
    end

    sumMismatch = 0;
    for i = 1:50
        sumMismatch = sumMismatch + abs(input(d+i) - input(floor(d+i*(p^(1/2)))));
    end
    result(ri) = (sumMismatch - sumMatch) / 100;
    ri = ri + 1;
end

figure
D = 1:499;
scatter(D,result);