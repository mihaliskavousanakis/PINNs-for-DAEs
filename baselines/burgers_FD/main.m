xpt=linspace(-6,6,6001); xpt=xpt'; 

u0=exp(-xpt.^2); u0(end+1)=0; u0(end+1)=0; 

template=exp(-xpt.^2); 
dtemplate=-2*xpt.*exp(-xpt.^2); 
dt=1e-2; 
t(1)=0; U(:,1)=u0; 
tic
while 1
    [u,flag]=burgers_rhs_forward(dt,u0,xpt,template,dtemplate);
    if flag==1
    break
    end
    U(:,end+1)=u; u0=u;
    t(end+1)=t(end)+dt;
    if t(end)>6
        break
    end
end
toc