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
ti1=linspace(0,0.2,11); ti1(1)=[];
ti2=linspace(0.2,1,11); ti2(1)=[];
ti3=linspace(1,4,11); ti3(1)=[];
ti4=linspace(4,8,11); ti4(1)=[];
ti5=linspace(8,12,11); ti5(1)=[];
ti=[ti1 ti2 ti3 ti4];

rho=linspace(0,4,101); rho(1)=[]; rho(end)=[];
theta=linspace(0,pi/2,11); theta(1)=[]; theta(end)=[];
drdth=(rho(2)-rho(1))*(theta(2)-theta(1));
[R,Th]=meshgrid(rho,theta); R=reshape(R,1,[]); Th=reshape(Th,1,[]);
R=repmat(R,1,length(ti)); Th=repmat(Th,1,length(ti));

T=repmat(ti,length(rho)*length(theta),1);T=reshape(T,1,[]);
CP(1,:)=T; CP(2,:)=R; CP(3,:)=Th;
CP=dlarray(CP,"CB");

% Boundary Conditions
% R=0, Th
[T,Th]=meshgrid(ti,theta);
T=reshape(T,1,[]); Th=reshape(Th,1,[]);
CB1(1,:)=T; 
CB1(2,:)=0; 
CB1(3,:)=Th; 
CB1=dlarray(CB1,"CB");

% R=10, Th
CB2(1,:)=T; 
CB2(2,:)=4; 
CB2(3,:)=Th; 
CB2=dlarray(CB2,"CB");

% R, Th=0
[T,R]=meshgrid(ti,rho);
T=reshape(T,1,[]); R=reshape(R,1,[]);
CB3(1,:)=T; 
CB3(2,:)=R; 
CB3(3,:)=0; 
CB3=dlarray(CB3,"CB");

% R, Th=pi/2;
CB4(1,:)=T; 
CB4(2,:)=R; 
CB4(3,:)=pi/2; 
CB4=dlarray(CB4,"CB");

CBt=ti; 
CBt=dlarray(CBt,"CB");

% Initial Conditions
[R,Th]=meshgrid(rho,theta);
R=reshape(R,1,[]);Th=reshape(Th,1,[]);
U0=1./(1+exp(-5.*(2-R)));
U0=dlarray(U0,"CB");
CBIC(1,:)=zeros(size(R));
CBIC(2,:)=R; 
CBIC(3,:)=Th; 
CBIC=dlarray(CBIC,"CB");

% For pinning condition
[T,Th]=meshgrid(ti,theta);
T=reshape(T,1,[]); Th=reshape(Th,1,[]);
CPin(1,:)=T; 
CPin(2,:)=0; 
CPin(3,:)=Th; 
CPin=dlarray(CPin,"CB");


parameters1=net1.Learnables;
parameters2=net2.Learnables;

parameters=cat(1,parameters1,parameters2);
accfun=dlaccelerate(@modelLoss);

load result_test
w1=1; w2=1; w3=1; w4=1; w5=1; w6=1; w7=1; w8=1;
numEpochs=15000; solverState=lbfgsState;

Loss=zeros(1,numEpochs);
lossFcn=@(parameters) dlfeval(accfun,parameters,net1,net2,CP,CB1,CB2,CB3,CB4,CBIC,CBt,CPin,U0,drdth,w1,w2,w3,w4,w5,w6,w7,w8);
 for i = (1):(numEpochs)

    [parameters, solverState] = lbfgsupdate(parameters,lossFcn,solverState);
 fprintf("Iteration %d: Loss: %d\n",i, solverState.Loss);
 Loss(i)=solverState.Loss;
 end
