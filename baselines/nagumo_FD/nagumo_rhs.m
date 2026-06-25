function dudt = nagumo_rhs(t,u,dx,alpha,xpt,template,templatex)

np=length(u)-1; 
dudt=zeros(np+1,1);

% node 1
i=1; 
dudt(i)=(u(i+1)-u(i))/dx^2+u(i).*(1-u(i)).*(u(i)-alpha);

% internal nodes
i=2:np-1; 
dudt(i)=(u(i+1)+u(i-1)-2*u(i))/dx^2+u(i).*(1-u(i)).*(u(i)-alpha)-u(end)*(u(i+1)-u(i-1))/(2*dx);

% node np
i=np; 
dudt(i)=(u(i-1)-u(i))/dx^2+u(i).*(1-u(i)).*(u(i)-alpha);

dudt(np+1)=trapz(xpt,(u(1:end-1)-template).*templatex);

