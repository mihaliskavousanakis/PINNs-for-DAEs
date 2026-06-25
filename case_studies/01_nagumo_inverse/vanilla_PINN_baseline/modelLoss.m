function [loss,gradients,mseF1,mseF2,mseF3,mseF4,mseF5,f1,f2,f3,f4]=modelLoss(parameters,net1,CP,CB,CBIC,U0,dataInfer,w1,w2,w3,w4,w5)

net1.Learnables=parameters(1:6,:); 
alpha=parameters(end,:).Value{1}; 

% make predictions
U=forward(net1,CP); 
gradientsU=dlgradient(sum(U,"all"),CP,EnableHigherDerivatives=true);
Ut=gradientsU(1,:);
Ux=gradientsU(2,:);
Uxx=dlgradient(sum(Ux,"all"),CP); Uxx=Uxx(2,:);

f1=Ut-alpha.*Uxx-U.*(1-U).*(U-0.01);
zeroTarget=zeros(size(f1),"like",f1);
mseF1 = l2loss(f1,zeroTarget);

% Boundary Conditions #1s
U1=forward(net1,CB);
gradientsU=dlgradient(sum(U1,"all"),CB);
Ux=gradientsU(2,:); 
f2=Ux;
zeroTarget=zeros(size(f2),"like",f2);
mseF2 = l2loss(f2,zeroTarget);

% Initial Condiiton
U0pred=forward(net1,CBIC);
mseF4 = l2loss(U0pred,U0);

% data
load data_for_Nagumo_fifth

Udata=forward(net1,dataInfer);
mseF6=mean((Udata-A(:,2)').^2,'all');



loss=w1*mseF1+w2*mseF2+w4*mseF4+mseF6;


gradients=dlgradient(loss,cat(1,parameters));


end
