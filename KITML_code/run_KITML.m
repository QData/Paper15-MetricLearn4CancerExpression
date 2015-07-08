%%
clear all; close all; clc;
dnames={'alizadehv1','bittner','bredel','garber','golubv1','golubv2','gordon',...
    'nuttv2','pomeroyv2','su','tomlinsv1','tomlinsv2','yeohv1','yeohv2'};
%%
%% choose 1 of the 14 data sets to run
K=1;
%%
datas=strcat(dnames{K},'_data')
labels=strcat(dnames{K},'_label')
tmpd=load(['data/' datas]);
X=tmpd.(datas);
tmpl=load(['data/' labels]);
y=tmpl.(labels);
%%
RN=3;
num_folds = 4;
knn_neighbor_size=[1:5 7 9 11];
SN=length( knn_neighbor_size);
accl=zeros(RN,SN);
Fmicrol=zeros(RN,SN);
Fmacrol=zeros(RN,SN);
f1accl=zeros(RN,SN);


for k= 1:1:length(knn_neighbor_size) 
    disp(['Neighbor size:' num2str(knn_neighbor_size(k))]);
    for r=1:RN % run
    [accl(r,k), predl, actl] = CrossValidateKNN_kernel(y, X, @(y,X) MetricLearningAutotuneKnn_kernel(@ItmlAlg_kernel, y, X), num_folds, knn_neighbor_size(k));
    [Fmicrol(r,k), Fmacrol(r,k),f1accl(r,k)] = f1score(predl, actl);
    end

end
results=[mean(accl); std(accl);  mean(Fmacrol); std(Fmacrol)]
