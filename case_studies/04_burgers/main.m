numLayers = 4; 
numNeurons=20; 

layers=featureInputLayer(2);

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
numNeurons=4; 

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
ti1=linspace(0,1,51);ti1(1)=[]; 
ti2=linspace(1,2,51); ti2(1)=[]; 
ti3=linspace(2,4,101); ti3(1)=[];
ti4=linspace(4,6,101); ti4(1)=[];
 ti=[ti1 ti2 ti3 ti4];
xpt=linspace(-6,6,601); xpt(1)=[]; xpt(end)=[];
[T,X]=meshgrid(ti,xpt);
T=reshape(T,1,300*599); X=reshape(X,1,300*599); 
CP(1,:)=T; CP(2,:)=X; 
CP=dlarray(CP,"CB");

% Boundary Conditions
CB1(1,:)=ti; 
CB1(2,:)=-6*ones(size(ti)); 
CB1=dlarray(CB1,"CB");

CB2(1,:)=ti; 
CB2(2,:)=6*ones(size(ti)); 
CB2=dlarray(CB2,"CB");

CBt=ti; 
CBt=dlarray(CBt,"CB");

% Initial Conditions 
xpt=linspace(-6,6,601);
U0=exp(-xpt.^2);
U0=dlarray(U0,"CB");

CBIC(1,:)=zeros(size(xpt));
CBIC(2,:)=xpt; 
CBIC=dlarray(CBIC,"CB");


w1=1; w2=1; w3=1; w4=1; w5=1; w6=1;

parameters1=net1.Learnables;
parameters2=net2.Learnables;

parameters=cat(1,parameters1,parameters2);
accfun=dlaccelerate(@modelLoss);
 w1=1; w2=1; w3=1; w4=1; w5=1; w6=1;
 numEpochs=5000; solverState=lbfgsState('HistorySize',50);
tic
lossFcn=@(parameters) dlfeval(accfun,parameters,net1,net2,CP,CB1,CB2,CBIC,CBt,U0,w1,w2,w3,w4,w5,w6);
 for i = (1):(numEpochs)

 [parameters, solverState] = lbfgsupdate(parameters,lossFcn,solverState);
 fprintf("Iteration %d: Loss: %d\n",i, solverState.Loss);
 Loss(i)=solverState.Loss;
 if mod(i,250)<1e-4
     [loss,gradients,mseF1,mseF2,mseF3,mseF4,mseF5,mseF6]=dlfeval(@modelLoss,parameters,net1,net2,CP,CB1,CB2,CBIC,CBt,U0,w1,w2,w3,w4,w5,w6);
     w1=w1+1.5*mseF1;
     w2=w2+1.5*mseF2;
     w3=w3+1.5*mseF3;
     w4=w4+1.5*mseF4;
     w5=w5+1.5*mseF5;
     w6=w6+1.5*mseF6;
 end
 end
