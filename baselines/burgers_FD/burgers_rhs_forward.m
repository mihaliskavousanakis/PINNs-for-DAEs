function [u,flag]=burgers_rhs_forward(dt,u0,xpt,template,dtemplate)


np=length(xpt); 
dx=xpt(2)-xpt(1); 
flag=0;
u=u0; 
nu=0.001;
while 1
    res=zeros(np+2,1);
    ajac=sparse(np+2,np+2);

    i=1;
    res(i)=u(np+1).*u(i) - (nu.*(2.*u(i) - 2.*u(i+1)))./dx.^2-(u(i)-u0(i))/dt;
    ajac=ajac+sparse(i,i,-1/dt+u(np+1) - (2.*nu)./dx.^2,np+2,np+2);
    ajac=ajac+sparse(i,i+1,(2.*nu)./dx.^2,np+2,np+2);
    ajac=ajac+sparse(i,np+1,u(i),np+2,np+2);
       
    
    i=2:np-1;
    res(i)=u(np+1).*(u(i) - (xpt(i).*(u(i-1) - u(i+1)))./(2.*dx)) - (u(np+2).*(u(i-1) - u(i+1)))./(2.*dx) + (u(i).*(u(i-1) - u(i+1)))./(2.*dx) + (nu.*(u(i-1) - 2.*u(i) + u(i+1)))./dx.^2-(u(i)-u0(i))/dt;
    ajac=ajac+sparse(i,i-1,nu./dx.^2 - u(np+2)./(2.*dx) + u(i)./(2.*dx) - (u(np+1).*xpt(i))./(2.*dx),np+2,np+2); 
    ajac=ajac+sparse(i,i,-1/dt+u(np+1) - (2.*nu)./dx.^2 + (u(i-1) - u(i+1))./(2.*dx),np+2,np+2);
    ajac=ajac+sparse(i,i+1,u(np+2)./(2.*dx) + nu./dx.^2 - u(i)./(2.*dx) + (u(np+1).*xpt(i))./(2.*dx),np+2,np+2);
    ajac=ajac+sparse(i,np+1,u(i) - (xpt(i).*(u(i-1) - u(i+1)))./(2.*dx),np+2,np+2);
    ajac=ajac+sparse(i,np+2,-(u(i-1) - u(i+1))./(2.*dx),np+2,np+2);
    
    i=np; 
    res(i)=u(np+1).*u(i) - (nu.*(2.*u(i) - 2.*u(i-1)))./dx.^2-(u(i)-u0(i))/dt;
    ajac=ajac+sparse(i,i-1,(2.*nu)./dx.^2,np+2,np+2);
    ajac=ajac+sparse(i,i,-1/dt+u(np+1) - (2.*nu)./dx.^2,np+2,np+2);
    ajac=ajac+sparse(i,np+1,u(i),np+2,np+2);

    i=2:np-1;

    res(np+1)=sum((u(i)-template(i)).*(xpt(i).*dtemplate(i)+template(i)))*dx;
    ajac=ajac+sparse(np+1,i,(xpt(i).*dtemplate(i)+template(i))*dx,np+2,np+2);

    res(np+2)=sum((u(i)-template(i)).*dtemplate(i))*dx;
    ajac=ajac+sparse(np+2,i,dtemplate(i)*dx,np+2,np+2); 

    du=ajac\(-res); 
    u=u+du; 
    max(abs(du));
    if max(abs(du))<1e-6
        break
    end
    if max(abs(du))>100
        flag=1; 
        return
    end
end
    


