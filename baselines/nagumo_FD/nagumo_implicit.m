function u=nagumo_implicit(u,xpt,dt,alpha,template,templatex);

dx=xpt(2)-xpt(1);
np=length(u)-1; 
dudt=zeros(np+1,1);

u0=u; 

while 1
    ajac=sparse(np+1,np+1);
    res=zeros(np+1,1); 

    % node 1
    i=1; 
    res(i)=(u(i+1)-u(i))/dx^2+u(i).*(1-u(i)).*(u(i)-alpha)-(u(i)-u0(i))/dt;
    ajac=ajac+sparse(i,i+1,1/dx^2,np+1,np+1);
    ajac=ajac+sparse(i,i,-1/dx^2+2.*u(i) - alpha + 2.*alpha.*u(i) - 3.*u(i).^2-1/dt,np+1,np+1);
    
    % internal nodes
    i=2:np-1; 
    res(i)=(u(i+1)+u(i-1)-2*u(i))/dx^2+u(i).*(1-u(i)).*(u(i)-alpha)-u(np+1)*(u(i+1)-u(i-1))/(2*dx)-(u(i)-u0(i))/dt;
    ajac=ajac+sparse(i,i+1,1/dx^2-u(np+1)/(2*dx),np+1,np+1);
    ajac=ajac+sparse(i,i,-2/dx^2+2.*u(i) - alpha + 2.*alpha.*u(i) - 3.*u(i).^2-1/dt,np+1,np+1);
    ajac=ajac+sparse(i,i-1,1/dx^2+u(np+1)/(2*dx),np+1,np+1);
    ajac=ajac+sparse(i,np+1,-(u(i+1)-u(i-1))/(2*dx),np+1,np+1);

    % node np
    i=np; 
    res(i)=(u(i-1)-u(i))/dx^2+u(i).*(1-u(i)).*(u(i)-alpha)-(u(i)-u0(i))/dt;
    ajac=ajac+sparse(i,i-1,1/dx^2,np+1,np+1);
    ajac=ajac+sparse(i,i,-1/dx^2+2.*u(i) - alpha + 2.*alpha.*u(i) - 3.*u(i).^2-1/dt,np+1,np+1);

    res(np+1)=sum((u(1:end-1)-template).*templatex)*dx;
    ajac=ajac+sparse(np+1,1:np,templatex(1:np)*dx,np+1,np+1);

    du=ajac\(-res);
    u=u+du;

    err=max(abs(du));
    if err<=1e-6
        break
    end


end