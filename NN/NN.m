% Neuron weights
w = [4 -2];
% Neuron bias
b = -3;
% Activation function: Hyperbolic tangent sigmoid function
func = 'tansig';
% Activation function: Logistic sigmoid transfer function
 func = 'logsig'
% Activation function: Hard-limit transfer function (threshold)
 func = 'hardlim'
% Activation function: Linear transfer function
 func = 'purelin' 
p = [2 3];
activation_potential = p*w'+b;
neuron_output = feval(func, activation_potential);
[p1,p2] = meshgrid(-10:.25:10);
z = feval(func, [p1(:) p2(:)]*w'+b );
z = reshape(z,length(p1),length(p2));
plot3(p1,p2,z);
grid on;
xlabel('Input 1');
ylabel('Input 2');
zlabel('Neuron output'); 
%% Custom NN
inputs = [1:6]; % input vector (6-dimensional pattern)
outputs = [7:12]; % corresponding target output vector 
% create the network: 1 input, 2 layer (1 hidden layer and 1 output layer), feed-forward
network
net = network( ...
1, ... % numInputs (number of inputs)
2, ... % numLayers (number of layers)
[1; 0], ... % biasConnect (numLayers-by-1 Boolean vector)
[1; 0], ... % inputConnect (numLayers-by-numInputs Boolean matrix)
[0 0; 1 0], ... % layerConnect (numLayers-by-numLayers Boolean matrix); [a b; c d]
... % a: 1st-layer with itself, b: 2nd-layer with 1st-layer,
... % c: 1st-layer with 2nd-layer, d: 2nd-layer with itself
[0 1] ... % outputConnect (1-by-numLayers Boolean vector)
);
% number of hidden layer neurons
net.layers{1}.size = 5;
% hidden layer transfer function
net.layers{1}.transferFcn = 'logsig';
net = configure(net,inputs,outputs);
view(net);
% initial network response without training (the network is resimulated)
initial_output = net(inputs) 
net.IW{1}
net.LW{2}
net.b{1}
% network training
net.trainFcn = 'trainlm'; % trainlm: Levenberg-Marquardt backpropagation
% trainlm is often the fastest backpropagation algorithm in the toolbox,
% and is highly recommended as a first choice supervised algorithm,
% although it does require more memory than other algorithms.
net.performFcn = 'mse';
net = train(net,inputs,outputs);
% network response after training (the network is resimulated)
final_output = net(inputs)
% final weight matrices and bias vector:
net.IW{1}
net.LW{2}
net.b{1}
%% Demo 11
nnd11gn
%% Perceptron
% number of samples of each class
N = 20;
% define inputs and outputs
offset = 5; % offset for second class
x = [randn(2,N) randn(2,N)+offset]; % inputs
y = [zeros(1,N) ones(1,N)]; % outputs
% Plot input samples with plotpv (Plot perceptron input/target vectors)
figure(1)
plotpv(x,y);
net = perceptron;
net = train(net, x, y);
view(net);
figure(1)
plotpc(net.IW{1},net.b{1});
% Plot a classification line on a perceptron vector plot 
%% 4 class perceptron
close all, clear all, clc
% number of samples of each class
K = 30;
% define clases
q = .6; % offset of clases
A = [rand(1,K)-q; rand(1,K)+q];
B = [rand(1,K)+q; rand(1,K)+q];
C = [rand(1,K)+q; rand(1,K)-q];
D = [rand(1,K)-q; rand(1,K)-q];
% plot clases
plot(A(1,:),A(2,:),'bs')
hold on
grid on
plot(B(1,:),B(2,:),'r+')
plot(C(1,:),C(2,:),'go')
plot(D(1,:),D(2,:),'m*')
% text labels for clases
text(.5-q,.5+2*q,'Class A')
text(.5+q,.5+2*q,'Class B')
text(.5+q,.5-2*q,'Class C')
text(.5-q,.5-2*q,'Class D')
% define output coding for clases
a = [0 1]';
b = [1 1]';
c = [1 0]';
d = [0 0]'; 
% define inputs (combine samples from all four classes)
P = [A B C D];
% define targets
T = [repmat(a,1,length(A)) repmat(b,1,length(B)) ... % repmat: Replicate and tile an array
repmat(c,1,length(C)) repmat(d,1,length(D)) ];
plotpv(P,T);
net = perceptron;
% To see the adaptation you need to look at the plot while the code is running
E = 1;
net.adaptParam.passes = 1;
linehandle = plotpc(net.IW{1},net.b{1});
n = 0;
while (sse(E) & n<1000) % sse: Sum squared error
    n = n+1;
 [net,Y,E] = adapt(net,P,T);
 linehandle = plotpc(net.IW{1},net.b{1},linehandle);
 drawnow;
end
% show perceptron structure
view(net); 
% For example, classify an input vector of [0.7; 1.2]
p = [0.7; 1.2]
y = net(p)
% compare response with output coding (a,b,c,d) 
%% Multilayer Perceptron
close all, clear all, clc
% number of samples of each class
K = 100;
% define 4 clusters of input data
q = .6; % offset of classes
A = [rand(1,K)-q; rand(1,K)+q];
B = [rand(1,K)+q; rand(1,K)+q];
C = [rand(1,K)+q; rand(1,K)-q];
D = [rand(1,K)-q; rand(1,K)-q];
% plot clusters
figure(1)
plot(A(1,:),A(2,:),'k+')
hold on
grid on
plot(B(1,:),B(2,:),'b*')
plot(C(1,:),C(2,:),'kx')
plot(D(1,:),D(2,:),'bd')
% text labels for clusters
text(.5-q,.5+2*q,'Class A')
text(.5+q,.5+2*q,'Class B')
text(.5+q,.5-2*q,'Class C') 
text(.5-q,.5-2*q,'Class D') 
% coding (+1/-1) of 4 separate classes
a = [-1 -1 -1 +1]';
b = [-1 -1 +1 -1]';
d = [-1 +1 -1 -1]';
c = [+1 -1 -1 -1]'; 
% define inputs (combine samples from all four classes)
P = [A B C D];
% define targets
T = [repmat(a,1,length(A)) repmat(b,1,length(B))...
repmat(c,1,length(C)) repmat(d,1,length(D))]; 
% create a neural network
net = feedforwardnet([4 3]); %Number of hidden neurons in each layer
% train net
net.divideParam.trainRatio = 1; % training set [%]
net.divideParam.valRatio = 0; % validation set [%]
net.divideParam.testRatio = 0; % test set [%]
% train a neural network
[net,tr,Y,E] = train(net,P,T); % Y:Output; E:Error
% show network
view(net) 
% evaluate performance: decoding network response
[m,i] = max(T); % target class
[m,j] = max(Y); % predicted class
N = length(Y); % number of all samples
k = 0; % number of missclassified samples
if find(i-j), % if there exist missclassified samples
 k = length(find(i-j)); % get a number of missclassified samples
end
fprintf('Correct classified samples: %.1f%% samples\n', 100*(N-k)/N)
% plot network output
figure;
subplot(211)
plot(T')
title('Targets')
ylim([-2 2])
grid on
subplot(212)
plot(Y')
title('Network response')
xlabel('# sample')
ylim([-2 2])
grid on 
% generate a grid
span = -1:.01:2;
[P1,P2] = meshgrid(span,span);
pp = [P1(:) P2(:)]';
% simualte neural network on a grid
aa = net(pp);
% plot classification regions based on MAX activation
figure(1)
m = mesh(P1,P2,reshape(aa(1,:),length(span),length(span))-5);
set(m,'facecolor',[1 0.2 .7],'linestyle','none');
hold on
m = mesh(P1,P2,reshape(aa(2,:),length(span),length(span))-5); 
set(m,'facecolor',[1 1.0 0.5],'linestyle','none');
m = mesh(P1,P2,reshape(aa(3,:),length(span),length(span))-5);
set(m,'facecolor',[.4 1.0 0.9],'linestyle','none');
m = mesh(P1,P2,reshape(aa(4,:),length(span),length(span))-5);
set(m,'facecolor',[.3 .4 0.5],'linestyle','none');
view(2)
%% Multilayer XOR
close all, clear all, clc, format compact
% number of samples of each class
K = 100;
% define 4 clusters of input data
q = .6; % offset of classes
A = [rand(1,K)-q; rand(1,K)+q];
B = [rand(1,K)+q; rand(1,K)+q];
C = [rand(1,K)+q; rand(1,K)-q];
D = [rand(1,K)-q; rand(1,K)-q];
% plot clusters
figure(1)
plot(A(1,:),A(2,:),'k+')
hold on
grid on
plot(B(1,:),B(2,:),'bd')
plot(C(1,:),C(2,:),'k+')
plot(D(1,:),D(2,:),'bd')
% text labels for clusters
text(.5-q,.5+2*q,'Class A')
text(.5+q,.5+2*q,'Class B')
text(.5+q,.5-2*q,'Class A')
text(.5-q,.5-2*q,'Class B')
% encode clusters a and c as one class, and b and d as another class
a = -1; % a | b
c = -1; % -------
b = 1; % d | c
d = 1; % 
% define inputs (combine samples from all four classes)
P = [A B C D];
% define targets
T = [repmat(a,1,length(A)) repmat(b,1,length(B)) ...
repmat(c,1,length(C)) repmat(d,1,length(D)) ];
% view inputs |outputs
% [P' T'] 
% create a neural network
% The sizes of the network are set to 0. These sizes will automatically be configured to
% match particular data by train
net = feedforwardnet([5 3]);
% train net
net.divideParam.trainRatio = 1; % training set
net.divideParam.valRatio = 0; % validation set
net.divideParam.testRatio = 0; % test set
% train a neural network
[net,tr,Y,E] = train(net,P,T);
% show network
view(net) 
figure(2)
plot(T','linewidth',2)
hold on
plot(Y','r--')
grid on
legend('Targets','Network response','location','best')
ylim([-1.25 1.25]) 
% generate a grid
span = -1:.005:2;
[P1,P2] = meshgrid(span,span);
pp = [P1(:) P2(:)]';
% simulate neural network on a grid
aa = net(pp);
% translate output into [-1,1]
% aa = -1 + 2*(aa>0);
 % plot classification regions
figure(1)
mesh(P1,P2,reshape(aa,length(span),length(span))-5);
colormap cool
view(2) 
%% RBFN XOR
close all, clear all, clc
% number of samples of each cluster
K = 100;
% offset of clusters
q = .6;
% define 2 groups of input data
A = [rand(1,K)-q rand(1,K)+q;
rand(1,K)+q rand(1,K)-q]; 
B = [rand(1,K)+q rand(1,K)-q;
rand(1,K)+q rand(1,K)-q];
% plot data
plot(A(1,:),A(2,:),'k+',B(1,:),B(2,:),'b*')
grid on
hold on 
% coding (+1/-1) for 2-class XOR problem
a = -1;
b = 1;
% define inputs (combine samples from all four classes)
P = [A B];
% define targets
T = [repmat(a,1,length(A)) repmat(b,1,length(B))]; 
% NEWRB algorithm
% The following steps are repeated until the network's mean squared error
% falls below goal:
% 1. The network is simulated
% 2. The input vector with the greatest error is found
% 3. A radial base neuron is added with weights equal to that vector
% 4. The purelin layer weights are redesigned to minimize error
% choose a spread constant
%The larger spread is, the smoother the function approximation. Too large a spread means
%a lot of neurons are required to fit a fast-changing function. Too small a spread means
%many neurons are required to fit a smooth function, and the network might not generalize
%well. Call newrb with different spreads to find the best value for a given problem.
spread = 2;
% choose max number of neurons
K = 20;
% performance goal (SSE)
goal = 0;
% number of neurons to add between displays
Ki = 4;
% create a neural network
net = newrb(P,T,goal,spread,K,Ki);
% view network
view(net) 
% simulate RBFN on training data
Y = net(P);
% calculate [%] of correct classifications
correct = 100 * length(find(T.*Y > 0)) / length(T);
fprintf('\nSpread = %.2f\n',spread)
fprintf('Num of neurons = %d\n',net.layers{1}.size)
fprintf('Correct class = %.2f %%\n',correct)
% plot targets and network response 
figure;
plot(T')
hold on
grid on
plot(Y','r')
ylim([-2 2])
set(gca,'ytick',[-2 0 2])
legend('Targets','Network response')
xlabel('Sample No.')
% generate a grid
span = -1:.025:2;
[P1,P2] = meshgrid(span,span);
pp = [P1(:) P2(:)]';
% simualte neural network on a grid
aa = sim(net,pp);
% plot classification regions based on MAX activation
figure(1)
ma = mesh(P1,P2,reshape(-aa,length(span),length(span))-5);
mb = mesh(P1,P2,reshape( aa,length(span),length(span))-5);
set(ma,'facecolor',[1 0.2 .7],'linestyle','none');
set(mb,'facecolor',[1 1.0 .5],'linestyle','none');
view(2) 
plot(net.iw{1}(:,1),net.iw{1}(:,2),'gs') 