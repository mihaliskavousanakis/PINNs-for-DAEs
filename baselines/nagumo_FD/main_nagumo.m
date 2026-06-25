% Nagumo equation 
xpt=linspace(-30,30,3001); dx=xpt(2)-xpt(1); 
xpt=xpt'; 

np=length(xpt); 
u=zeros(np+1,1); 

u0=0.*(xpt<0)+xpt/10.*(xpt>=0).*(xpt<=10)+1*(xpt>10); u0(end+1)=0.5; 
template=u0(1:end-1); 
templatex=0.*(xpt<0)+1/10.*(xpt>=0).*(xpt<=10)+0*(xpt>10);

alpha=0.01;

tic
U(:,1)=u0;
dt=0.01; T(1)=0;
for met=1:20/0.01;
    U(:,end+1)=nagumo_implicit(u0,xpt,dt,alpha,template,templatex);
    u0=U(:,end);
    T(end+1)=T(end)+dt; 
end
toc


