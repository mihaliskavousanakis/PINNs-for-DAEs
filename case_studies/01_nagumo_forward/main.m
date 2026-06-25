numLayers = 3; 
numNeurons=40; 

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
numNeurons=5; 

layers=featureInputLayer(1);

for i=1:numLayers-1
    layers = [layers
        fullyConnectedLayer(numNeurons)
        tanhLayer];
end

layers = [layers
    fullyConnectedLayer(1)];

net2=dlnetwork(layers);
net2 = dlupdate(@double,net2);



% Collocation points
ti1=linspace(0,0.1,11);ti1(1)=[]; 
ti2=linspace(0.1,1,11);ti2(1)=[];
ti3=linspace(1,5,11);ti3(1)=[];
ti4=linspace(5,20,11);ti4(1)=[];
ti=[ti1 ti2 ti3 ti4];
xpt=linspace(-30,30,601); xpt(1)=[]; xpt(end)=[];
[T,X]=meshgrid(ti,xpt);
T=reshape(T,1,length(ti)*599); X=reshape(X,1,length(ti)*599); 
CP(1,:)=T; CP(2,:)=X; 
CP=dlarray(CP,"CB");

% Boundary Conditions
CB1(1,:)=ti; 
CB1(2,:)=-30*ones(size(ti)); 
CB1=dlarray(CB1,"CB");

CB2(1,:)=ti; 
CB2(2,:)=30*ones(size(ti)); 
CB2=dlarray(CB2,"CB");

CBt=ti; 
CBt=dlarray(CBt,"CB");

% Initial Conditions 
xpt=linspace(-30,30,601);
U0=0.*(xpt<0)+xpt/10.*(xpt>=0).*(xpt<=10)+1*(xpt>10);
U0=dlarray(U0,"CB");
Ux0=0.*(xpt<0)+1/10.*(xpt>=0).*(xpt<=10)+0*(xpt>10);
Ux0=dlarray(Ux0,"CB");
CBIC(1,:)=zeros(size(xpt));
CBIC(2,:)=xpt; 
CBIC=dlarray(CBIC,"CB");


w1=1; w2=1; w3=1; w4=1; w5=1; 

parameters1=net1.Learnables;
parameters2=net2.Learnables;

parameters=cat(1,parameters1,parameters2);
accfun=dlaccelerate(@modelLoss);
lossFcn=@(parameters) dlfeval(accfun,parameters,net1,net2,CP,CB1,CB2,CBIC,CBt,U0,Ux0,w1,w2,w3,w4,w5);

load result_1

w1=1; w2=1; w3=1; w4=1; w5=1;
 numEpochs=1000; solverState=lbfgsState;
tic
 for i = 1:numEpochs
 [parameters, solverState] = lbfgsupdate(parameters,lossFcn,solverState);
 fprintf("Iteration %d: Loss: %d\n",i, solverState.Loss);
 
 
 if mod(i,250)<1e-4
     [loss,gradients,mseF1,mseF2,mseF3,mseF4,mseF5]=dlfeval(@modelLoss,parameters,net1,net2,CP,CB1,CB2,CBIC,CBt,U0,Ux0,w1,w2,w3,w4,w5);
     w1=w1+2^(i/250)*mseF1;
     w2=w2+2^(i/250)*mseF2;
     w3=w3+2^(i/250)*mseF3;
     w4=w4+2^(i/250)*mseF4;
     w5=w5+2^(i/250)*mseF5;     
 end
 if solverState.Loss < 1e-6
     break
 end
 end
toc