numLayers = 4; 
numNeurons=40; 

layers=featureInputLayer(3);

for i=1:numLayers-1
    layers = [layers
        fullyConnectedLayer(numNeurons)
        tanhLayer];
end

layers = [layers
    fullyConnectedLayer(1)];

net1=dlnetwork(layers);
net1 = dlupdate(@double,net1);

numLayers = 2; 
numNeurons=5; 

layers=featureInputLayer(1);

for i=1:numLayers-1
    layers = [layers
        fullyConnectedLayer(numNeurons)
        tanhLayer];
end

layers = [layers
    fullyConnectedLayer(2)];

net2=dlnetwork(layers);
net2 = dlupdate(@double,net2);



% Collocation points
load result_ti5
ti=tinew;
xpt=linspace(0,4,41); xpt(1)=[]; xpt(end)=[];
ypt=linspace(0,4,41); ypt(1)=[]; ypt(end)=[];
[X,Y]=meshgrid(xpt,ypt); X=reshape(X,1,39*39); Y=reshape(Y,1,39*39);
X=repmat(X,1,40); Y=repmat(Y,1,40);

T=repmat(ti,39*39,1);T=reshape(T,1,40*39*39);
CP(1,:)=T; CP(2,:)=X; CP(3,:)=Y;
CP=dlarray(CP,"CB");

% Boundary Conditions
% x=0, y
[T,Y]=meshgrid(ti,ypt);
T=reshape(T,1,39*40); Y=reshape(Y,1,39*40);
CB1(1,:)=T; 
CB1(2,:)=0; 
CB1(3,:)=Y; 
CB1=dlarray(CB1,"CB");

% x=4, y
CB2(1,:)=T; 
CB2(2,:)=4; 
CB2(3,:)=Y; 
CB2=dlarray(CB2,"CB");

% x, y=0
[T,X]=meshgrid(ti,xpt);
T=reshape(T,1,39*40); X=reshape(X,1,39*40);
CB3(1,:)=T; 
CB3(2,:)=X; 
CB3(3,:)=0; 
CB3=dlarray(CB3,"CB");

% x, y=4;
CB4(1,:)=T; 
CB4(2,:)=X; 
CB4(3,:)=4; 
CB4=dlarray(CB4,"CB");

CBt=ti; 
CBt=dlarray(CBt,"CB");

% Initial Conditions
xpt=linspace(0,4,41);ypt=linspace(0,4,41);
[X,Y]=meshgrid(xpt,ypt);
X=reshape(X,1,41*41);Y=reshape(Y,1,41*41);
U0=exp(-X-Y);
U0=dlarray(U0,"CB");

CBIC(1,:)=zeros(size(X));
CBIC(2,:)=X; 
CBIC(3,:)=Y; 
CBIC=dlarray(CBIC,"CB");


parameters1=net1.Learnables;
parameters2=net2.Learnables;

parameters=cat(1,parameters1,parameters2);
accfun=dlaccelerate(@modelLoss);

load result_ti5
ti=tinew;
 numEpochs=5000; solverState=lbfgsState;
tic
return
 for i = (1):(numEpochs)
     lossFcn=@(parameters) dlfeval(accfun,parameters,net1,net2,CP,CB1,CB2,CB3,CB4,CBIC,CBt,U0,w1,w2,w3,w4,w5,w6,w7,w8);
    [parameters, solverState] = lbfgsupdate(parameters,lossFcn,solverState);
 fprintf("Iteration %d: Loss: %d\n",i, solverState.Loss);
 Loss(i)=solverState.Loss;
 if mod(i,50)<1e-4
     [loss,gradients,mseF1,mseF2,mseF3,mseF4,mseF5,mseF6,mseF7,mseF8]=dlfeval(@modelLoss,parameters,net1,net2,CP,CB1,CB2,CB3,CB4,CBIC,CBt,U0,w1,w2,w3,w4,w5,w6,w7,w8);
     w1=w1+15*mseF1;
     w2=w2+15*mseF2;
     w3=w3+15*mseF3;
     w4=w4+15*mseF4;
     w5=w5+15*mseF5;
     w6=w6+15*mseF6;
     w7=w7+15*mseF7;
     w8=w8+15*mseF8;
 end
 end
