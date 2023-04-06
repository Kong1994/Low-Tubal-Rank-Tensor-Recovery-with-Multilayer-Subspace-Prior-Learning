clear
addpath(genpath(cd))

load news.mat
%% 训练数据生成1
X1 =news(:,:,:,1:40);
X1=imresize(X1,0.5);
 W=zeros(72,88,3,40);
[n1,n2,n3,n4] = size(X1);
Y=zeros([n1 n2 n3*n4]);
X1=reshape(X1,size(Y));
X1 = X1/255;
Xtrain1 = X1;
%% 测试数据的生成
XT=news(:,:,:,41:60);
XT=imresize(XT,0.5);
H=zeros(72,88,3,20);
[m1,m2,m3,m4] = size(XT);
Z=zeros([m1 m2 m3*m4]);
XT=reshape(XT,size(Z));
XT = XT/255;
Xtest = XT;
opts.mu = 1e-4;
opts.mu_bar = 1e10;
opts.tol = 1e-6;
opts.rho = 1.5;
opts.maxIter = 500;
opts.DEBUG =1;
%% 产生先验子空间信息
%% training
DataTrain1=Xtrain1;
XTEST=Xtest;
[A1,B1,A2,B2,k1,k2,k3,XTEST]=subspace2tpcp(XTEST,DataTrain1);
%% add noise
[b1,b2,b3] = size(XTEST);
        img_Test = XTEST;
        rhos = 0.1;
        ind = find(rand(b1*b2*b3,1)<rhos);
        img_Test(ind) = rand(length(ind),1);
        M=img_Test;
lambda = 4/sqrt(k3*max(k1,k2)); 
%% IMTPCP
tic
[Xhat_MTPCP,Ehat,err] =trpcam2(M,A1,B1,A2,B2,lambda,opts);
Mtime=toc
maxP = max(abs(XTEST(:)));
Xhat_MTPCP = max(Xhat_MTPCP,0);
Xhat_MTPCP = min(Xhat_MTPCP,maxP);
psnr_MTPCP = PSNR(XTEST,Xhat_MTPCP,maxP);
[ssim_MTPCP] = ssim(Xhat_MTPCP,XTEST);
fsim_MTPCP=FeatureSIM(Xhat_MTPCP,XTEST);
Erroralm = norm(Xhat_MTPCP(:)-XTEST(:))/norm(XTEST(:));
fprintf('Relative error = %0.8e\n',Erroralm);
D_MTPCP=Xhat_MTPCP(:)-XTEST(:);
MSE_MTPCP= sum(D_MTPCP(:).*D_MTPCP(:))/numel(Xhat_MTPCP(:));
MAE_MTPCP=mean(mean(abs(D_MTPCP)));%平均绝对误差
fprintf('psnr = %0.8e\n',psnr_MTPCP);
fprintf('ssim = %0.8e\n',ssim_MTPCP);
 