function u=burgers_rhs(dt,u0,xpt,template,dtemplate)


np=length(xpt); 
dx=xpt(2)-xpt(1); 

u=u0; 

while 1
    res=zeros(np+2,1); 
    ajac=sparse(np+2,np+2); 

    i=1;
    res(i)=0.01.*(2.*u(i+1)-2.*u(i))./dx.^2+u(np+1).*u(i)-(u(i)-u0(i))/dt;
    ajac=ajac+sparse(i,i,-1/dt+u(np+1) - 1./(20.*dx.^2),np+2,np+2); 
    ajac=ajac+sparse(i,i+1,1./(20.*dx.^2),np+2,np+2);
    ajac=ajac+sparse(i,np+1,u(i),np+2,np+2);
    
    
    i=2:np-1;
    res(i)=0.01.*(u(i+1)+u(i-1)-2.*u(i))./dx.^2-u(i).*(u(i+1)-u(i-1))./(2.*dx)+u(np+1).*(u(i)+xpt(i).*(u(i+1)-u(i-1))./(2.*dx))+u(np+2).*(u(i+1)-u(i-1))./(2.*dx)-(u(i)-u0(i))/dt;
    ajac=ajac+sparse(i,i-1,u(i)./(2.*dx) - u(np+2)./(2.*dx) + 1./(40.*dx.^2) - (u(np+1).*xpt(i))./(2.*dx),np+2,np+2); 
    ajac=ajac+sparse(i,i,-1/dt+u(np+1) - 1./(20.*dx.^2) + (u(i-1) - u(i+1))./(2.*dx),np+2,np+2);
    ajac=ajac+sparse(i,i+1,u(np+2)./(2.*dx) - u(i)./(2.*dx) + 1./(40.*dx.^2) + (u(np+1).*xpt(i))./(2.*dx),np+2,np+2);
    ajac=ajac+sparse(i,np+1,u(i) - (xpt(i).*(u(i-1) - u(i+1)))./(2.*dx),np+2,np+2);
    ajac=ajac+sparse(i,np+2,-(u(i-1) - u(i+1))./(2.*dx),np+2,np+2);

    i=np; 
    res(i)=0.01.*(2.*u(i-1)-2.*u(i))./dx.^2+u(np+1).*u(i)-(u(i)-u0(i))/dt;
    ajac=ajac+sparse(i,i-1,1./(20.*dx.^2),np+2,np+2); 
    ajac=ajac+sparse(i,i,-1/dt+u(np+1) - 1./(20.*dx.^2),np+2,np+2);
    ajac=ajac+sparse(i,np+1,u(i),np+2,np+2);

    i=2:np-1;

    res(np+1)=sum((u(i)-template(i)).*(xpt(i).*dtemplate(i)+template(i)))*dx;
    ajac=ajac+sparse(np+1,i,(xpt(i).*dtemplate(i)+template(i))*dx,np+2,np+2);

    res(np+2)=sum((u(i)-template(i)).*dtemplate(i))*dx;
    ajac=ajac+sparse(np+2,i,dtemplate(i)*dx,np+2,np+2); 

    du=ajac\(-res); 
    u=u+du; 
max(abs(du))
    if max(abs(du))<1e-6
        break
    end
end
    


