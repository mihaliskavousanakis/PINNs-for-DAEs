function [loss,gradients,mseF1,mseF2,mseF3,mseF4,mseF5,mseF6]=modelLoss(parameters,net1,net2,CP,CB1,CB2,CBIC,CBt,U0,w1,w2,w3,w4,w5,w6,Y,Temp,dx)

net1.Learnables=parameters(1:8,:); net2.Learnables=parameters(9:end,:);
% make predictions
U=forward(net1,CP); AB=forward(net2,CBt);
gradientsU=dlgradient(sum(U,"all"),CP,EnableHigherDerivatives=true);
Ut=gradientsU(1,:);
Ux=gradientsU(2,:);
Uxx=dlgradient(sum(Ux,"all"),CP,EnableHigherDerivatives=true); Uxx=Uxx(2,:);

A=repmat(AB(1,:),499,1); A=reshape(A,1,499*100);
B=repmat(AB(2,:),499,1); B=reshape(B,1,499*100);
f1=(Ut-2*Uxx.*U-2*Ux.^2+A.*(U)-B.*Ux.*CP(2,:))-2*U.*Ux./CP(2,:);
zeroTarget=zeros(size(f1),"like",f1);
mseF1 = l2loss(f1,zeroTarget);

% Boundary Conditions #1
U1=forward(net1,CB1);
f2=U1;
zeroTarget=zeros(size(f2),"like",f2);
mseF2 = l2loss(f2,zeroTarget);

% Boundary Conditions #2
U2=forward(net1,CB2);
gradientsU=dlgradient(sum(U2,"all"),CB2,EnableHigherDerivatives=true);
Ux=gradientsU(2,:); 
f3=Ux;
zeroTarget=zeros(size(f3),"like",f3);
mseF3 = l2loss(f3,zeroTarget);

% Initial Condiiton
U0pred=forward(net1,CBIC);
mseF4 = l2loss(U0pred,U0);

% Template Condition
U=reshape(U,499,100); U=U';

f4=dx*sum((U).*(Temp),2)'; f4=dlarray(f4,"CB");
zeroTarget=zeros(size(f4),"like",f4);
mseF5 = l2loss(f4,zeroTarget);

f5=U2-1;
zeroTarget=zeros(size(f5),"like",f5);
mseF6 = l2loss(f5,zeroTarget);


loss=w1*mseF1+w2*mseF2+w3*mseF3+w4*mseF4+w5*mseF5+w6*mseF6;

gradients=dlgradient(loss,cat(1,net1.Learnables,net2.Learnables));


end
