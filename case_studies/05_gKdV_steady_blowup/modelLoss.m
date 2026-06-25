    
function [loss,gradients,mseF1,mseF2,mseF3,mseF4,mseF5]=modelLoss(parameters,net1,CP,CB1,CB2,xpin,w1,w2,w3,w4,w5)

net1.Learnables=parameters(1:end-1,:);
Gi=parameters(end,:).Value{1};
% make predictions
U=forward(net1,CP);
Ux=dlgradient(sum(U,"all"),CP,EnableHigherDerivatives=true); 
Uxx=dlgradient(sum(Ux,"all"),CP,EnableHigherDerivatives=true); 
Uxxx=dlgradient(sum(Uxx,"all"),CP,EnableHigherDerivatives=true); 

p=5.2; 
f1=Uxxx+p*abs(U).^(p-1).*Ux-Gi.*(2.*U/(p-1)+CP.*Ux)-Ux;
mseF1=mean(f1.^2,'all'); 

% Boundary conditions 
% x=-L
U1=forward(net1,CB1);
Ux=dlgradient(sum(U1,"all"),CB1,EnableHigherDerivatives=true);
f2=2*Gi/(p-1).*U1+(Gi.*CB1+1).*Ux;
mseF2 = mean(f2.^2,'all');

% x=L
U2=forward(net1,CB2);
Ux=dlgradient(sum(U2,"all"),CB2,EnableHigherDerivatives=true);
Uxx=dlgradient(sum(Ux,"all"),CB2,EnableHigherDerivatives=true); 
f3=2*Gi/(p-1).*U2+(Gi.*CB2+1).*Ux;
mseF3 = mean(f3.^2,'all');
f4=U2; 
mseF4 = mean(f4.^2,'all');

% Pinning condition
Upin=forward(net1,xpin);
f5=Upin-1;
mseF5 = mean(f5.^2,'all');

loss=w1*mseF1+w2*mseF2+w3*mseF3+w4*mseF4+w5*mseF5;

gradients=dlgradient(loss,parameters);


end
