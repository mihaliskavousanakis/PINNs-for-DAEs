function [u,flag]=pme2d_rhs(dt,u0,xpt,template)


np=length(xpt); 
dx=xpt(2)-xpt(1); 

u=u0; 
flag=0; 
while 1
    res=zeros(np+2,1); 
    ajac=sparse(np+2,np+2); 

     i=1;
    res(i)=u(i);
    ajac=ajac+sparse(i,i,1,np+2,np+2); 
    
    i=2:np-1;
    res(i)=(u(i+1).^2+u(i-1).^2-2.*u(i).^2)./dx.^2.*xpt(i)+(u(i+1).^2-u(i-1).^2)./(2.*dx)-xpt(i).*u(np+1).*u(i)+xpt(i).^2.*u(np+2).*(u(i+1)-u(i-1))./(2.*dx)-(u(i)-u0(i))/dt.*xpt(i);
    ajac=ajac+sparse(i,i-1,(2.*u(i-1).*xpt(i))./dx.^2 - u(i-1)./dx - (u(np+2).*xpt(i).^2)./(2.*dx),np+2,np+2); 
    ajac=ajac+sparse(i,i,-xpt(i)/dt- u(np+1).*xpt(i) - (4.*u(i).*xpt(i))./dx.^2,np+2,np+2);
    ajac=ajac+sparse(i,i+1,u(i+1)./dx + (2.*u(i+1).*xpt(i))./dx.^2 + (u(np+2).*xpt(i).^2)./(2.*dx),np+2,np+2);
    ajac=ajac+sparse(i,np+1,-u(i).*xpt(i),np+2,np+2); 
    ajac=ajac+sparse(i,np+2,-(xpt(i).^2.*(u(i-1) - u(i+1)))./(2.*dx),np+2,np+2);

    i=np; 
    res(i)=(2.*u(i-1).^2-2.*u(i).^2)./dx.^2.*xpt(i)-xpt(i).*u(np+1).*u(i)-(u(i)-u0(i))/dt.*xpt(i);
    ajac=ajac+sparse(i,i-1,(4.*u(i-1).*xpt(i))./dx.^2,np+2,np+2); 
    ajac=ajac+sparse(i,i,-xpt(i)/dt- u(np+1).*xpt(i) - (4.*u(i).*xpt(i))./dx.^2,np+2,np+2);
    ajac=ajac+sparse(i,np+1,-u(i).*xpt(i),np+2,np+2); 

    i=2:np-1; 
    res(np+1)=sum(u(i).*template(i))*dx;
    ajac=ajac+sparse(np+1,i,(template(i))*dx,np+2,np+2);

    res(np+2)=u(np)-1;
    ajac=ajac+sparse(np+2,np,1,np+2,np+2); 

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
    


