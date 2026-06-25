function [loss,gradients,mseF1,mseF2,mseF3,mseF4,mseF5,mseF6,mseF7,mseF8]=modelLoss(parameters,net1,net2,CP,CB1,CB2,CB3,CB4,CBIC,CBt,CPin,U0,drdth,w1,w2,w3,w4,w5,w6,w7,w8)

net1.Learnables=parameters(1:8,:); net2.Learnables=parameters(9:end,:);
% make predictions
U=forward(net1,CP); AB=forward(net2,CBt);
gradU = dlgradient(sum(U, "all"), CP, EnableHigherDerivatives=true);
[Ut, Ur, Uth] = deal(gradU(1,:), gradU(2,:), gradU(3,:));
Urr=dlgradient(sum(Ur,"all"),CP,EnableHigherDerivatives=true); Urr=Urr(2,:);
Uthth=dlgradient(sum(Uth,"all"),CP,EnableHigherDerivatives=true); Uthth=Uthth(3,:);

AB_exp = reshape(repmat(AB, length(CBIC), 1), 2, []);
A = AB_exp(1,:); B = AB_exp(2,:);
f1=Ut-2*(Ur.^2+(Uth./CP(2,:)).^2)-2*U.*(Urr+Uthth./CP(2,:).^2+Ur./CP(2,:))-A.*CP(2,:).*Ur+B.*U;

zeroTarget=zeros(size(f1),"like",f1);
mseF1 = l2loss(f1,zeroTarget);

% Boundary Conditions #1 - R=0
U1=forward(net1,CB1);
gradientsU=dlgradient(sum(U1,"all"),CB1,EnableHigherDerivatives=true);
Ur=gradientsU(2,:); 
f2=Ur;
zeroTarget=zeros(size(f2),"like",f2);
mseF2 = l2loss(f2,zeroTarget);

% Boundary Conditions #2 - R=10
U2=forward(net1,CB2);
f3=U2;
zeroTarget=zeros(size(f3),"like",f3);
mseF3 = l2loss(f3,zeroTarget);

% Boundary Conditions #3 - Th=0
U3=forward(net1,CB3);
gradientsU=dlgradient(sum(U3,"all"),CB3,EnableHigherDerivatives=true);
Uth=gradientsU(3,:); 
f4=Uth;
zeroTarget=zeros(size(f4),"like",f4);
mseF4 = l2loss(f4,zeroTarget);

% Boundary Conditions #4 - Th=pi/2
U4=forward(net1,CB4);
gradientsU=dlgradient(sum(U4,"all"),CB4,EnableHigherDerivatives=true);
Uth=gradientsU(3,:); 
f5=Uth;
zeroTarget=zeros(size(f5),"like",f5);
mseF5 = l2loss(f5,zeroTarget);

% Initial Condiiton
U0pred=forward(net1,CBIC);
mseF6 = l2loss(U0pred,U0);

% Pinning Condition #1
U=reshape(U,length(CBIC),[]); U=U';
Temp=1-2./(1+exp(-5*(CP(2,:)-1)));
Temp=reshape(Temp,length(CBIC),[]); Temp=Temp';

f7=drdth*sum(U.*Temp,2)'; f7=dlarray(f7,"CB");
zeroTarget=zeros(size(f7),"like",f7);
mseF7 = l2loss(f7,zeroTarget);

% Pinning Condition #2
Upin=forward(net1,CPin);
f8=Upin-1;
zeroTarget=zeros(size(f8),"like",f8);
mseF8 = l2loss(f8,zeroTarget);

loss=w1*mseF1+w2*mseF2+w3*mseF3+w4*mseF4+w5*mseF5+w6*mseF6+w7*mseF7+w8*mseF8;

gradients=dlgradient(loss,cat(1,net1.Learnables,net2.Learnables));

end
