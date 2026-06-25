function [loss,gradients,mseF1,mseF2,mseF3,mseF4,mseF5,f1,f2,f3,f4]=modelLoss(parameters,net1,net2,CP,CB,CBIC,CBt,U0,Ux0,dataVel,A,w1,w2,w3,w4,w5)

net1.Learnables=parameters(1:6,:); net2.Learnables=parameters(7:end,:);
alpha=parameters(end,:).Value{1}; 

% make predictions
U=forward(net1,CP); vel=forward(net2,CBt);
gradientsU=dlgradient(sum(U,"all"),CP,EnableHigherDerivatives=true);
Ut=gradientsU(1,:);
Ux=gradientsU(2,:);
Uxx=dlgradient(sum(Ux,"all"),CP); Uxx=Uxx(2,:);


V=repmat(vel,299,1); V=reshape(V,1,299*300);
f1=Ut-(alpha).*Uxx-U.*(1-U).*(U-0.01)-V.*Ux;
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


% Template Condition (interior x; do not mutate U0/Ux0 used for mseF4)
U0s=U0(2:end-1);
Ux0s=Ux0(2:end-1);
U=reshape(U,299,300); U=U';
U0=repmat(U0s,300,1);
Ux0=repmat(Ux0s,300,1);
dx=CP(2,2)-CP(2,1);
f4=dx.*sum((U-U0).*Ux0,2)';
zeroTarget=zeros(size(f4),"like",f4);
mseF5 = l2loss(f4,zeroTarget);

% data
Veldata=forward(net2,dataVel);
Totshift=zeros(size(A(1,:)),"like",Veldata);
dataInferShifted=zeros(size(A),"like",A);
j1=1;
for i=2:size(A(2,:),2)
    dt_i=dataVel(1,i)-dataVel(1,i-1);
    jend=(i-1)*10;
    s=0; 
    for j=(j1+1):jend
        s=s+Veldata(1,j-1).*dt_i;
    end
    Totshift(1,i)=Totshift(1,i-1)+s;
    j1=jend;
end
dataInferShifted=A(1,:);
dataInferShifted(2,:)=-10-Totshift;
Wdata=forward(net1,dataInferShifted);

mseF6=mean((Wdata-A(2,:)).^2,'all');



loss=w1*mseF1+w2*mseF2+w4*mseF4+w5*mseF5+mseF6;

gradients=dlgradient(loss,cat(1,parameters));


end
