function [loss,gradients,mseF1,mseF2,mseF3,mseF4,mseF5]=modelLoss(parameters,net1,net2,CP,CB1,CB2,CBIC,CBt,U0,Ux0,w1,w2,w3,w4,w5)

net1.Learnables=parameters(1:6,:); net2.Learnables=parameters(7:end,:);
% make predictions
U=forward(net1,CP); vel=forward(net2,CBt);
gradientsU=dlgradient(sum(U,"all"),CP,EnableHigherDerivatives=true);
Ut=gradientsU(1,:);
Ux=gradientsU(2,:);
Uxx=dlgradient(sum(Ux,"all"),CP,EnableHigherDerivatives=true); Uxx=Uxx(2,:);

V=repmat(vel,599,1); V=reshape(V,1,599*40);
f1=Ut-Uxx-U.*(1-U).*(U-0.01)+V.*Ux;
zeroTarget=zeros(size(f1),"like",f1);
mseF1 = l2loss(f1,zeroTarget);

% Boundary Conditions #1
U1=forward(net1,CB1);
gradientsU=dlgradient(sum(U1,"all"),CB1,EnableHigherDerivatives=true);
Ux=gradientsU(2,:); 
f2=Ux;
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
U0(1)=[]; Ux0(1)=[];
U0(end)=[]; Ux0(end)=[];
U=reshape(U,599,40); U=U';
U0=repmat(U0,40,1);
Ux0=repmat(Ux0,40,1);
dx=CP(2,2)-CP(2,1);
f4=extractdata(dx)*sum((U-U0).*Ux0,2)'; f4=dlarray(f4,"CB");
zeroTarget=zeros(size(f4),"like",f4);
mseF5 = l2loss(f4,zeroTarget);


loss=w1*mseF1+w2*mseF2+w3*mseF3+w4*mseF4+w5*mseF5;

gradients=dlgradient(loss,cat(1,net1.Learnables,net2.Learnables));


end
