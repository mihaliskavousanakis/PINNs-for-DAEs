numLayers = 3; 
numNeurons=40; 

layers=featureInputLayer(1);

for i=1:numLayers-1
    layers = [layers
        fullyConnectedLayer(numNeurons)
        tanhLayer];
end

layers = [layers
    fullyConnectedLayer(1)];

net1=dlnetwork(layers);
net1 = dlupdate(@double,net1);



% Collocation points
xpt=linspace(-20,20,2001);
CP(1,:)=xpt;
CP=dlarray(CP,"CB");

% Boundary Conditions
CB1=min(xpt); 
CB1=dlarray(CB1,"CB");

CB2=max(xpt); 
CB2=dlarray(CB2,"CB");

% Pinning condition
xpin=dlarray(0,"CB"); 


w1=1; w2=1; w3=1; w4=1; w5=1; w6=1;

parameters=net1.Learnables;

structtest(1,1).Layer="Gcoeff";
structtest(1,1).Parameter="pinparameter";
structtest(1,1).Value={dlarray(-1,"CB")};

parameters=[parameters;struct2table(structtest)];
accfun=dlaccelerate(@modelLoss);

 numEpochs=1;
learnRate=1e-4;
gradDecay=0.9;
sqGradDecay=0.999;
trailingAvg = [];
trailingAvgSq = [];
tic
 iend=0;
 w1=1; w2=1; w3=1;
 numEpochs=15000; solverState=lbfgsState;
 load initial_weights_gaussian

tic
lossFcn=@(parameters) dlfeval(accfun,parameters,net1,CP,CB1,CB2,xpin,w1,w2,w3,w4,w5);
for i = (1+iend):(numEpochs+iend)

 [parameters, solverState] = lbfgsupdate(parameters,lossFcn,solverState);
 if mod(i,50)<1e-4
 fprintf("Iteration %d: Loss: %d G= %d \n",i, solverState.Loss,parameters(end,:).Value{1});

 end
 Loss(i)=solverState.Loss;
 if Loss(i)<1e-9
     break
 end
 end
toc