clear;
clc
addpath datasets\yaleA_3view\
load ('yaleA_3view_0.5_baladata')
param.lamda1 = 0.005;
param.lamda2 = 0.01;
param.lamda3 =100;
param.lamda4 = 0.01;
param.beta = 1;
param.Maxiter = 200;
truthF = gt;
num_cluster = length(unique(truthF));
N=length(truthF);
num_view = length(X);
options = [];
options.NeighborMode = 'KNN';
options.k =7;
options.WeightMode = 'Binary';
m1=30;m2=30;m3=3;repeat=30;

for iv=1:num_view
    X2 = X{iv};
    ind_0 = find(inds(:,iv)==0);
    X2(:,ind_0) = 0;
    XX{iv} = X2;
    Win1=eye(N);
    ind_1=find(inds(:,iv)==1);
    Win1(ind_1,:)=[];
    P{iv}=Win1;
    Ind_ms{iv}=ind_0;
end
clear X2 Win1 iv

S2=zeros(N);
for iv = 1:num_view
    S1 = constructW(X{iv}',options);%第iv个视图的邻接矩阵
    S1 = full(S1);%将稀疏矩阵转为全矩阵
    S1 = S1 - diag(diag(S1));  % to make the diagonal elements being  zeros
    S1 = (S1+S1)*0.5;%对称化
    S2 = S2 + S1;%累加各个视图的邻接矩阵
end
S = (S2+S2)*0.5;
sumS = max(1e-15,full(sum(S,1)));
S = S*diag(sumS.^-1);  % to make the column sums being ones
clear S1 S2
for iz=1:num_view
    Z{iz}=S;
end
ZTD=cat(3,Z{:,:});
Z_Tensor=tensor(ZTD);

U=nvecs(Z_Tensor,1,m1);
V=nvecs(Z_Tensor,2,m2);
W=nvecs(Z_Tensor,3,m3);
G_Tensor=ttm(Z_Tensor,{U',V',W'});
U(U<0)=0;
V(V<0)=0;
W(W<0)=0;
G=double(G_Tensor);
G(G<0)=0;
G_Tensor=tensor(G);

[G_Tensor1,V1,U1,Objective,tempVG2,err,err1_values,diffZD_values,diffZC_values]=Update_model_without_Graph_Laplace(XX,G_Tensor,Z_Tensor,Z,U,V,W,param,S,P,N,num_view);
G_2=tenmat(G_Tensor1,2);
F=V1*G_2.data;
F2=[U1,V1];
cunchu_result=zeros(repeat,7);
cunchu_result_plus=zeros(repeat,7);
for ip=1:repeat
    [preLabels_km,Clustering_centroid,distant_sum,disTT] = kmeans(F,num_cluster,'emptyaction','singleton','replicates',20,'display','off');
    preLabels_km_plus=kmeans(F, num_cluster, 'Start', 'plus', 'MaxIter', 100, 'Replicates', 20);
    result =100* EvaluationMetrics(truthF,preLabels_km);
    result_plus =100* EvaluationMetrics(truthF,preLabels_km_plus);
    preLabels_km2 = kmeans(F2,num_cluster,'emptyaction','singleton','replicates',20,'display','off');
    result2 = EvaluationMetrics(truthF,preLabels_km);
    cunchu_result(ip, :) = result;
    cunchu_result_plus(ip, :) = result_plus;
    clear result2 preLabels_km2  result_plus result preLabels_km_plus
end
kms_age = mean(cunchu_result);
kms_plus_aconstructWge = mean(cunchu_result_plus);
std_Kmeans = std(cunchu_result);
dist_to_centroid = disTT(sub2ind(size(disTT), (1:size(disTT, 1))', preLabels_km));
disTT =  (disTT - min(disTT(:))) / (max(disTT(:)) - min(disTT(:)));
%相似矩阵对角图
% Simar = constructW(F);
% imagesc(Simar)
% n11 = size(preLabels_km, 1);  % 数据点个数
% k11 = max(preLabels_km);    % 聚类数量
% IndicatorMatrix = full(sparse(1:n11, preLabels_km, 1, n11, k11));
%散点图
% Y = tsne(F);
% gscatter(Y(:,1), Y(:,2),truthF);   
% h = gscatter(Y(:,1), Y(:,2), preLabels_km_plus, 'cmykbrg', 'xo*^sdpvH+<>',2);
% figure;
% hold on;
% 迭代误差图
% Plot with different markers for each line and set MarkerSize
% figure;
% hold on
% plot(err1_values, 'r-', 'DisplayName', '||X^{(i)}+A^{(i)}P^{(i)}-(X^{(i)}+A^{(i)}P^{(i)})Z^{(i)} - E^{(i)}||_{\infty}');
% plot(diffZD_values, 'g-', 'DisplayName', '||Z^{(i)} - D^{(i)}||_{\infty}');
% plot(diffZC_values, 'b-', 'DisplayName', '||Z^{(i)} - C^{(i)}||_{\infty}');
% hold off
% xlabel('Iteration');
% ylabel('Error');
% legend('show');
% grid on;
% % 这里是 局部放大图 用于残差局部
% zp = BaseZoom();
% zp.plot;