
clear
clc
%% Setting the hyper-parameters
choose_norm=2; % Normalization methods, 0: no normalization, 1: z-score, 2: max-min
init=4; % Initialization methods, 1: random, 2: K-means, 3: fuzzt c-means, 4: K-means clustering, accelerated by matlab matrix operations.
repeat_num=10; % Repeat the experiment repeat_num times
addpath(genpath('.'));

%% load data
load('data.mat');
data=data(:, 2:end);
real_label=data(:, 1);
K=length(unique(real_label)); 
[N, ~]=size(data);
label_old=zeros(N, repeat_num);
s_1=0; 
%% Initialization & Normalization
 data = normlization(data, choose_norm);
for i=1:repeat_num
    label_old(:, i)=init_methods(data, K, init);
end
%% Repeat the experiment repeat_num times
t0=cputime;
for i=1:repeat_num
    [label_new, iter_GMM, responsivity,Class,para_miu, para_sigma,para_pi]=GMM_kailugaji(data, K, label_old(:, i));
    iter_GMM_t(i)=iter_GMM;
    %% performanc indices
     [accuracy(i), RI(i), NMI(i)]=performance_index(real_label,label_new);
     s_1=s_1+iter_GMM_t(i);
     fprintf('Iteration %2d, the number of iterations: %2d, Accuary: %.8f\n', i, iter_GMM_t(i), accuracy(i));
end
run_time=cputime-t0;
W1=responsivity(1:314,1);
W2=responsivity(358:543,2);
%% ConfusionMatrix    
T2ConfMat=confusionmat(real_label,Class1);
error12=(real_label==1)&(Class1==2);
error13=(real_label==1)&(Class1==3);
error14=(real_label==1)&(Class1==4);

error23=(real_label==2)&(Class1==3);
error24=(real_label==2)&(Class1==4);

error31=(real_label==3)&(Class1==1);
error32=(real_label==3)&(Class1==2);
error34=(real_label==3)&(Class1==4);

error41=(real_label==4)&(Class1==1);
error43=(real_label==4)&(Class1==3);

%% 
[~,Class1]=max(responsivity1,[],2);

figure;
gscatter(data(:,2),data(:,1),real_label,['r','k','b','g'],'.',12)
hold on
h=gscatter(data(:,2),data(:,1),Class1,['r','k','b','g'],'o',6);
hold on
legend(h,'Sample cluster 1','Sample cluster 2','Sample cluster 3','Sample cluster 4')
hold off
xlabel('Kurtosis') 
ylabel('Standard deviation') 

