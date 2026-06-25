function [loss,gradients,mseF1,mseF2,mseF3,mseF4,mseF5,mseF6,mseF7,mseF8]=modelLoss(parameters,net1,net2,CP,CB1,CB2,CB3,CB4,CBIC,CBt,U0,w1,w2,w3,w4,w5,w6,w7,w8)

net1.Learnables=parameters(1:8,:); net2.Learnables=parameters(9:end,:);
% make predictions
U=forward(net1,CP); AB=forward(net2,CBt);
gradU = dlgradient(sum(U, "all"), CP, EnableHigherDerivatives=true);
[Ut, Ux, Uy] = deal(gradU(1,:), gradU(2,:), gradU(3,:));
Uxx=dlgradient(sum(Ux,"all"),CP,EnableHigherDerivatives=true); Uxx=Uxx(2,:);
Uyy=dlgradient(sum(Uy,"all"),CP,EnableHigherDerivatives=true); Uyy=Uyy(3,:);

AB_exp = reshape(repmat(AB, 39*39, 1), 2, []);
A = AB_exp(1,:); B = AB_exp(2,:);
f1=Ut-Uxx-Uyy-A.*(CP(2,:).*Ux+CP(3,:).*Uy)+B.*U;
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
f3=U2;
zeroTarget=zeros(size(f3),"like",f3);
mseF3 = l2loss(f3,zeroTarget);

% Boundary Conditions #3
U3=forward(net1,CB3);
gradientsU=dlgradient(sum(U3,"all"),CB3,EnableHigherDerivatives=true);
Uy=gradientsU(3,:); 
f4=Uy;
zeroTarget=zeros(size(f4),"like",f4);
mseF4 = l2loss(f4,zeroTarget);

% Boundary Conditions #4
U4=forward(net1,CB4);
f5=U4;
zeroTarget=zeros(size(f5),"like",f5);
mseF5 = l2loss(f5,zeroTarget);
% Initial Condiiton
U0pred=forward(net1,CBIC);
mseF6 = l2loss(U0pred,U0);
% Template Condition
U=reshape(U,39*39,40); U=U';

Temp=exp(-CP(2,:)-CP(3,:)); 
dTempX=-Temp;
dTempY=-Temp;
Temp=reshape(Temp,39*39,40); Temp=Temp'; 
dTempX=reshape(dTempX,39*39,40); dTempX=dTempX'; 
dTempY=reshape(dTempY,39*39,40); dTempY=dTempY';

dx = extractdata(CP(3,2) - CP(3,1));
intWeight = dx * dx;

dx=CP(3,2)-CP(3,1); 
diffU=U-Temp;
f6=intWeight*sum(diffU.*Temp,2)'; f6=dlarray(f6,"CB");
zeroTarget=zeros(size(f6),"like",f6);
mseF7 = l2loss(f6,zeroTarget);
Xnew=CP(2,:); Xnew=reshape(Xnew,39*39,40); Xnew=Xnew'; 
Ynew=CP(3,:); Ynew=reshape(Ynew,39*39,40); Ynew=Ynew'; 
f7=intWeight*sum(diffU.*(dTempX.*Xnew+dTempY.*Ynew),2)'; f7=dlarray(f7,"CB");
zeroTarget=zeros(size(f7),"like",f7);
mseF8 = l2loss(f7,zeroTarget);

loss=w1*mseF1+w2*mseF2+w3*mseF3+w4*mseF4+w5*mseF5+w6*mseF6+w7*mseF7+w8*mseF8;

gradients=dlgradient(loss,cat(1,net1.Learnables,net2.Learnables));

end
