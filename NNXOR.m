%implement a 2-layer Neural Network (MLP) deal with XOR


%2,3,1 neurons in 3 layer, with one extra bias unit in layer 0 and 1 (bias unit may not be necessary)
% layer 0 --input, layer 1--hidden, layer 2 -- ouput(only one neuron)

% a=f(W*x+b), f(.) is sigmf()


%Training -- use vectorized batch gradient descent, from Andrew Ng's course
% fuction -- XOR
% 0 0 | 0
% 0 1 | 1
% 1 0 | 1
% 1 1 | 0


clear all
% Taining samples
T0=[0 0 0;
0 1 1;
1 0 1;
1 1 0];

% T0=[-1 -1 -1;
% -1 1 1;
% 1 -1 1;
% 1 1 -1];

%NN parameter

%W0 -- layer 0 weights
%W1 -- layer 1 weights
%b0 -- layer 0 bias
%b1 -- layer 1 bias
%similarly define a0 a1 a2, a0 is input

%init -- important!!
%W(i,j) means weight from (layer n-1, neruon j) to (layer n, neruon i) 

%initialize weight matrix with normal distribution N(0,1)
%initialize bias vector with normal distribution N(0,1)
sig=1
W0=normrnd(0,sig, [3,2]), %3x2
W1=normrnd(0,sig, [1,3]), %1x3
b0=zeros(3,1);
b1=0;

%learning rate -- important!! 
alpha=1;

for i=1:5000 %epochs, termination condition: 5000 epoch or Loss function < 0.01
    
    Loss(i)=0; %Loss function - define as (sum(y-a2))^2, square error between network output and correct answers.
    dw0=zeros(3,2);
    db0=zeros(3,1);
    dw1=zeros(1,3);
    db1=0;
    for j=1:4 %4 training samples
        %get activated output and Loss
        a0=T0(j,1:2)';
        a1=activate(W0,a0,b0);
        a2=activate(W1,a1,b1);
        d=T0(j,3)-a2; % error
        Loss(i)=Loss(i)+0.25*d.^2; %Loss function evaluate and accumulate, 0.25=1/4
        
        
        %training --backprop
        
        %layer 2
        der2=a2*(1-a2); %derivative of output layer / 1-d
        err2=-d.*der2; %error of output layer / 1-d
        dw1=dw1+err2*a1'; % accumulate weight changing / 1x3
        db1=db1+err2; %accumulate bias changing / 1-d
        
        %layer 1 --prop error back
        der1=a1.*(1-a1); %3x1
        err1=(W1'*err2).*der1; %3x1
        dw0=dw0+err1*a0'; %err1-3x1, a0-2x1, this one is 3x2!! 
        db0=db0+err1; %3x1
        
        %layer 0 is input, end
        
        
    end
    
    %update weight:
    W1=W1-(alpha*0.25).*dw1;
    W0=W0-(alpha*0.25).*dw0;
    b1=b1-(alpha*0.25).*db1;
    b0=b0-(alpha*0.25).*db0;
        
        
        
    
    
    if Loss<0.01
        break
    end
end

%Test:
%% Loss function show convergence

figure 
plot(Loss)
title('Loss function')
%% 

for k=1:4 %4 test cases
        %get activated output 
        a0=T0(k,1:2)';
        a1=activate(W0,a0,b0);
        test_r(k)=activate(W1,a1,b1),
        
end



