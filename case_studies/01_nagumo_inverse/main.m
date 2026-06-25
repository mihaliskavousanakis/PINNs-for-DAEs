numLayers = 3; 
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
ti=linspace(0,30,301);ti(1)=[]; 
xpt=linspace(-30,30,301); xpt(1)=[]; xpt(end)=[];
[T,X]=meshgrid(ti,xpt);
T=reshape(T,1,300*299); X=reshape(X,1,300*299); 
CP(1,:)=T; CP(2,:)=X; 
CP=dlarray(CP,"CB");

% Boundary Conditions
CB1(1,:)=ti; 
CB1(2,:)=-30*ones(size(ti)); 
CB1=dlarray(CB1,"CB");

CB2(1,:)=ti; 
CB2(2,:)=30*ones(size(ti)); 
CB2=dlarray(CB2,"CB");

CB=[CB1 CB2];

CBt=ti; 
CBt=dlarray(CBt,"CB");

% Initial Conditions
xpt=linspace(-30,30,301);
k =2;                     % steepness factor (adjust as needed)
U0 = 1 ./ (1 + exp(-k*(xpt - 5)));
U0=dlarray(U0,"CB");
Ux0=(k*exp(-k*(xpt - 5)))./(exp(-k*(xpt - 5)) + 1).^2;
Ux0=dlarray(Ux0,"CB");
CBIC(1,:)=zeros(size(xpt));
CBIC(2,:)=xpt; 
CBIC=dlarray(CBIC,"CB");

load data_for_Nagumo_fifth.mat
ti1=[];
for j=2:length(A(:,1))
    if ~isempty(ti1)
        ti1(end)=[];
    end
    ti1=[ti1 linspace(A(j-1,1),A(j,1),11)];
end
dataInfer(1,:)=ti1; 
dataInfer(2,:)=-10; 
dataInfer=dlarray(dataInfer,'CB'); 
dataVel(1,:)=dataInfer(1,:); 
dataVel=dlarray(dataVel,'CB'); 
t0=dlarray(0,'CB');
A=dlarray(A','CB');

w1=1; w2=1; w3=1; w4=1; w5=1; 

parameters1=net1.Learnables;

parameters2=net2.Learnables;
structtest(1,1).Layer="alpha";
structtest(1,1).Parameter="alpha";
structtest(1,1).Value={dlarray(5,"CB")};
parameters2=[parameters2;struct2table(structtest)];

parameters=cat(1,parameters1,parameters2);

accfun=dlaccelerate(@modelLoss);

load result_test

numEpochs=5000; solverState=lbfgsState;
W1=[]; W2=[]; W3=[]; W4=[]; W5=[]; E=[]; Loss=zeros(1,numEpochs);
lossFcn=@(parameters) dlfeval(accfun,parameters,net1,net2,CP,CB,CBIC,CBt,U0,Ux0,dataVel,A,w1,w2,w3,w4,w5);
 for i = 1:numEpochs

 [parameters, solverState] = lbfgsupdate(parameters,lossFcn,solverState);

 fprintf("Iteration %d: Loss: %d\n",i, solverState.Loss);

 Loss(i)=solverState.Loss;
 end
