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
    fullyConnectedLayer(2)];

net2=dlnetwork(layers);
net2 = dlupdate(@double,net2);



% Collocation points
ti=linspace(0,5,101);ti(1)=[]; 
xpt=linspace(-8,8,401); xpt(1)=[]; xpt(end)=[];
[T,X]=meshgrid(ti,xpt);
T=reshape(T,1,100*399); X=reshape(X,1,100*399); 
CP(1,:)=T; CP(2,:)=X; 
CP=dlarray(CP,"CB");

% Boundary Conditions
CB1(1,:)=ti; 
CB1(2,:)=-8*ones(size(ti)); 
CB1=dlarray(CB1,"CB");

CB2(1,:)=ti; 
CB2(2,:)=8*ones(size(ti)); 
CB2=dlarray(CB2,"CB");

CBt=ti; 
CBt=dlarray(CBt,"CB");

% Initial Conditions 
xpt=linspace(-8,8,401);
U0=1/2*(tanh((xpt+1)/0.2)-tanh((xpt-1)/0.2));
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
 numEpochs=15000; solverState=lbfgsState;
tic
 for i = (1):(numEpochs)
     lossFcn=@(parameters) dlfeval(accfun,parameters,net1,net2,CP,CB1,CB2,CBIC,CBt,U0,w1,w2,w3,w4,w5,w6);
    [parameters, solverState] = lbfgsupdate(parameters,lossFcn,solverState);
 fprintf("Iteration %d: Loss: %d\n",i, solverState.Loss);
 Loss(i)=solverState.Loss;
 if mod(i,100)<1e-4
     [loss,gradients,mseF1,mseF2,mseF3,mseF4,mseF5,mseF6]=dlfeval(@modelLoss,parameters,net1,net2,CP,CB1,CB2,CBIC,CBt,U0,w1,w2,w3,w4,w5,w6);
     w1=w1+2*mseF1;
     w2=w2+2*mseF2;
     w3=w3+2*mseF3;
     w4=w4+2*mseF4;
     w5=w5+2*mseF5;
     w6=w6+2*mseF6;
 end
 end

 save result_15000_for_paper
