function [G_Tensor,V,U,Objective,tempVG2,err,err1_values,diffZD_values,diffZC_values]=Update_model(X,G_Tensor,Z_Tensor,Z,U,V,W,param,S,P,N,num_view)
lamda1=param.lamda1;
lamda2=param.lamda2;
lamda3=param.lamda3;
lamda4=param.lamda4;
Maxiter=param.Maxiter;
beta=param.beta;
% epson=1e-7;
S1=sum(S,1);
DL=diag(S1);
L = diag(sum(S,1))-(S+S')*0.5;
VV=V'*V;
WW=W'*W;
Objective=zeros(1,Maxiter);
err=zeros(1,Maxiter);
maxmu=10^50;
rho1=1.2;
rho2=1.2;
rho3=1.2;
mu1=1e-4;
mu2=1e-4;
mu3=1e-4;
tempVG2=cell(1,Maxiter);
err1_values = zeros(Maxiter, 1);
diffZD_values = zeros(Maxiter, 1);
diffZC_values = zeros(Maxiter, 1);
% converge_Z_G=[];converge_Z=[];
for ii=1:num_view
    Q1{ii}=zeros(size(X{ii},1),N);
    Q2{ii}=zeros(N);
    Q3{ii}=zeros(N);
    E{ii}=zeros(size(X{ii},1),N);
end
D_Tensor=Z_Tensor;
tempQ2=cat(3,Q2{:,:});
Q2_Tensor=tensor(tempQ2);
tempQ3=cat(3,Q3{:,:});
Q3_Tensor=tensor(tempQ3);
clear tempQ3 tempQ2 ii
iter=0;
while iter < Maxiter
    fprintf('----processing iter %d--------\n', iter+1);
    %% Update A(v)
    for ia=1:num_view
        Y_1=X{ia}-X{ia}*Z{ia}-E{ia}+Q1{ia}/mu1;
        Y_2=P{ia}-P{ia}*Z{ia};
        %使用矩阵的逆去求解
        tempA=-mu1*Y_1*Y_2';
        tempA1=inv(2*lamda3*eye(size(P{ia},1))+mu1*Y_2*Y_2');
        A{ia}=tempA*tempA1;
        % 使用sylvester方程去求解
        % tempAA1=2*lamda3*eye(size(X{ia},1));
        % tempAA2=mu1*Y_2*Y_2';
        % tempAA3=-mu1*Y_1*Y_2';
        % A{ia}=sylvester(tempAA1,tempAA2,tempAA3);
    end
    clear Y_1 Y_2 tempA tempA1 tempA2 ia tempAA1 tempAA2 tempAA3
    
    Asum=0;
    for i3=1:num_view
        Asum=Asum+lamda3*norm(A{i3},'fro')^2;
    end

    %% Update E
    for ie=1:num_view
        D1{ie}=X{ie}+A{ie}*P{ie}-(X{ie}+A{ie}*P{ie})*Z{ie}+Q1{ie}/mu1;%??????但是这里 D1并不为0  为什么E会为0
        E{ie}=solve_l1l2(D1{ie},lamda1/mu1);
    end
    clear D1 ie
    Esum=0;
    for in=1:num_view
        Esum=Esum+lamda1*sum(sqrt(sum(E{in}.^2)));
    end

    %% Update U
    ZVW=ttm(D_Tensor,{V',W'},[2,3]);
    ZVW1=tenmat(ZVW,1).data;
    G_1=tenmat(G_Tensor,1).data;
    tempU1=ZVW1*G_1';
    GUVVWW=ttm(G_Tensor,{U,VV,WW},[1,2,3]);
    GUVVWW1=tenmat(GUVVWW,1).data;
    tempU2=GUVVWW1*G_1';
    tempU3=beta*U*trace(WW);
    U=U.*tempU1./(tempU2+tempU3);
    % U(U<0)=0;
    UU=U'*U;
    clear ZVW ZVW1 tempU1 tempU2 tempU3

    %% Update W
    G_3=tenmat(G_Tensor,3).data;
    ZUV=ttm(D_Tensor,{U',V'},[1,2]);
    ZUV1=tenmat(ZUV,3).data;
    tempW1=ZUV1*G_3';
    GUUVVW=ttm(G_Tensor,{UU,VV,W},[1 2 3]);
    GUUVVW1=tenmat(GUUVVW,3).data;
    tempW2=GUUVVW1*G_3';
    tempW3=beta*W*trace(UU);
    W=W.*tempW1./(tempW2+tempW3);
    % W(W<0)=0;
    WW=W'*W;
    clear ZUV ZUV1 tempW2 tempW3 tempW1

    %% Update V
    G_2=tenmat(G_Tensor,2).data;
    ZUW=ttm(D_Tensor,{U',W'},[1,3]);
    ZUW1=tenmat(ZUW,2).data;
    tempV1=ZUW1*G_2';
    tempV2=lamda2*S*V*G_2*G_2';
    GUUVWW=ttm(G_Tensor,{UU,V,WW},[1,2,3]);
    GUUVWW1=tenmat(GUUVWW,2).data;
    tempV3=GUUVWW1*G_2';
    tempV4=lamda2*DL*V*G_2*G_2';
    V=V.*(tempV1+tempV2)./(tempV3+tempV4);
    % V(V<0)=0;
    VV=V'*V;
    clear tempV1 tempV2 tempV3 tempV4

    %% Update G_Tensor
    ZUVW=ttm(D_Tensor,{U',V',W'},[1,2,3]).data;
    VSV=V'*S*V;
    GVSV=lamda2*ttm(G_Tensor,{VSV},2).data;
    GUUVVWW=ttm(G_Tensor,{UU,VV,WW},[1,2,3]).data;
    VDV=V'*DL*V;
    GVDV=lamda2*ttm(G_Tensor,{VDV},2).data;
    GD=G_Tensor.data;
    G_Tensor=GD.*(ZUVW+GVSV)./(GUUVVWW+GVDV);
    % G_Tensor(G_Tensor<0);
    G_Tensor=tensor(G_Tensor);
    clear VSV GVSV VDV GVDV GD


    %% UPdate C_Tensor
    ZTD=Z_Tensor.data;
    Q2TD=Q2_Tensor.data;
    ZTDv=ZTD(:);
    Q2TDv=Q2TD(:);
    [Cv,Kernelsum]=wshrinkObj(ZTDv+1/mu2*Q2TDv,lamda4/mu2,[N,N,num_view],0,3);
    C_TensorD=reshape(Cv,[N,N,num_view]);
    for ic=1:num_view
        C{ic}=C_TensorD(:,:,ic);
    end
    C_Tensor=tensor(C_TensorD);%这里的C张量是0 换句话说C可能并没有得到更新
    clear ZTD Q2TD ZTDv Q2TDv ic

    %% Update D Tensor
    Y_3=ttm(G_Tensor,{U,V,W},[1,2,3]).data;
    Y_3D=Y_3(:);
    ZTD=Z_Tensor.data(:);
    Q3D=Q3_Tensor.data(:);
    DD=1/(2+mu3)*(2*Y_3D+mu3*ZTD+Q3D);
    D_TensorD=reshape(DD,[N,N,num_view]);
    for id=1:num_view
        D{id}=D_TensorD(:,:,id);
    end
    D_Tensor=tensor(D_TensorD);
    % Y_3_2=tenmat(Y_3,2).data;
    % Y_4=Z_Tensor+Q3_Tensor./mu3;
    % Y_4_2=tenmat(Y_4,2).data;
    % D_T_2=(1/(2+mu3)).*(2*Y_3_2+mu3*Y_4_2);
    % D_Tensor1=reshape(D_T_2,[N,N,num_view]);%这里没有写完
    % D_Tensor2=permute(D_Tensor1,[2,1,3]);
    % for id=1:num_view
    %     D{id}=D_Tensor2(:,:,id);
    % end
    % D_Tensor=tensor(D_Tensor2);
    clear Y_4 Y_4_2 Y_3 Y_3_2 ZTD Q3D Y_3D DD

    %% Update Z^(v)

    for i4=1:num_view
        Y_5{i4}=X{i4}+A{i4}*P{i4};
        tempZ=mu1*Y_5{i4}'*Y_5{i4};
        tempZ1=(mu2+mu3)*eye(N);
        tempZ2=mu1*Y_5{i4}'*Y_5{i4}-mu1*Y_5{i4}'*E{i4}+Y_5{i4}'*Q1{i4}+mu2*C{i4}-Q2{i4}+mu3*D{i4}-Q3{i4};
        tempZZ1=tempZ+tempZ1;
        % tempZ3=sylvester(tempZ,tempZ1,tempZ2);                                             
        tempZ3=tempZZ1\tempZ2;
        Z1=zeros(size(tempZ3,1));
        for is = 1:size(tempZ3,1)
            ind_c = 1:size(tempZ3,1);
            ind_c(is) = [];
            Z1(is,ind_c) = EProjSimplex_new(tempZ3(is,ind_c));
        end
        Z{i4}=Z1;
    end
        clear tempZ tempZ1 tempZ2 tempZ3 i4
    Z_TD=cat(3,Z{:,:});
    Z_Tensor=tensor(Z_TD);

    %% Update Q1,Q2,Q3

    for i5=1:num_view
        Q1{i5}=Q1{i5}+mu1*(X{i5}+A{i5}*P{i5}-(X{i5}+A{i5}*P{i5})*Z{i5}-E{i5});
    end

    for iq2=1:num_view
        Q2{iq2}=Q2{iq2}+mu2*(Z{iq2}-C{iq2});
    end
    Q2_TensorD=cat(3,Q2{:,:});
    Q2_Tensor=tensor(Q2_TensorD);

    for iq3=1:num_view
        Q3{iq3}=Q3{iq3}+mu3*(Z{iq3}-D{iq3});
    end
    Q3_TensorD=cat(3,Q3{:,:});
    Q3_Tensor=tensor(Q3_TensorD);
    clear Q2_TensorD Q3_TensorD

    %% Objective function value
    
    tempF=tenmat(G_Tensor,2).data;
    VG2=V*tempF;
    Trsum=lamda2*trace(VG2'*L*VG2);
    K=ttm(G_Tensor,{U,V,W},[1,2,3]).data;
    tempDK=D_Tensor.data-K;
    Tuckersum=norm(tempDK,'fro')^2;
    Objective(iter+1)=Esum+Asum+Tuckersum+Trsum+Kernelsum;
    clear tempF tempDK K VG2

    tempVG2_1=tenmat(G_Tensor,2).data;
    tempVG2{iter+1}=V*tempVG2_1;
    
    %% convergence condition
    err1=0;
    for ix=1:num_view
        XAP=X{ix}+A{ix}*P{ix};
        Rec_error=XAP-XAP*Z{ix}-E{ix};
        err1=max(err1,max(abs(Rec_error(:))));
    end
    diffZD=max(abs(Z_Tensor.data(:)-D_Tensor.data(:)));
    diffZC=max(abs(Z_Tensor.data(:)-C_Tensor.data(:)));
    err(iter+1)=max([err1,diffZD,diffZC]);
    err1_values(iter+1) = err1;
    diffZD_values(iter+1) = diffZD;
    diffZC_values(iter+1) = diffZC;
    if iter >3 &&  abs(Objective(iter)-Objective(iter-1))<1e-6 || err(iter+1)<1e-8
        %iter
        break;
    end
    fprintf('iter = %d, mu1 = %d,mu2 = %d,mu3 = %d\n,Objective = %d,err = %d, Esum = %f,Asum=%d,Tuckersum=%d,Trsum=%f,Kernelsum=%d\n,err1=%d,diffZD=%d,diffZC=%d\n'...
            , iter,mu1,mu2,mu3,Objective(iter+1),err(iter+1),Esum,Asum,Tuckersum,Trsum,Kernelsum,err1,diffZD,diffZC);
    iter=iter+1;
    %% Update mu
    mu1 = min(maxmu,mu1*rho1);
    mu2 = min(maxmu,mu2*rho2);
    mu3 = min(maxmu,mu3*rho3);
end
end

