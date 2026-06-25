xpt=linspace(0,10,5001); xpt=xpt'; 

u0=1 ./ (1 + exp(-2.5*(xpt - 4))); u0(end+1)=0; u0(end+1)=0; 
load init_9.mat
u0=u; 
template=-1 + 2 ./ (1 + exp(-2.5*(xpt - 7))); 

dt=1e-3; 
t(1)=9; U(:,1)=u0; 
tic
while 1
    [u,flag]=pme2d_rhs(dt,u0,xpt,template);
    if flag==1
        break
    end
    U(:,end+1)=u; u0=u;
    t(end+1)=t(end)+dt;
    if t(end)>10
        break
    end
end
toc