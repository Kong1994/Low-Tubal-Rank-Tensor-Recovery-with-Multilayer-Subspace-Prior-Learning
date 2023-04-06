clear
addpath(genpath(cd))

load yaleB01.mat
X=X/255;
maxP = max(abs(X(:)));
[n1,n2,n3] = size(X);
opts.mu = 1e-4;
opts.mu_bar = 1e10;
opts.tol = 1e-8;
opts.rho = 1.1;
opts.maxIter = 500;
opts.DEBUG =1;
 p =0.1;    
%% 产生先验子空间信息
%% training
tl=40;
DataTrain=(X(:,:,1:tl));
XT=X(:,:,tl+1:60);
[A1,B1,A2,B2,k1,k2,k3,XT]=subspace2tc(XT,DataTrain);
%%  Set  Omega0.1
fprintf('Sampling ratio = %0.8e\n',p);
temp = randperm(k1*k2*k3);
kks = round(p*k1*k2*k3);
omega=temp(1:kks);
mark = zeros(k1,k2,k3); 
mark(temp(1:kks)) = 1;
M=XT.*mark;
tic
[Xhat_M,err] =lrtc_tnnm2(M,omega ,A1,B1,A2,B2,opts);% for YaleB
toc
M=permute(M,shift_dim);
XT=permute(XT,shift_dim);
Xhat_M=permute(Xhat_M,shift_dim);
Xhat_M = max(Xhat_M,0);
Xhat_M = min(Xhat_M,maxP);
%% print the relative error, psnr,ssim
psnr_M = PSNR(XT,Xhat_M,maxP);
ssim_M = ssim(Xhat_M,XT);
fsim_M=FeatureSIM(Xhat_M,XT);
ErrorsM= norm(Xhat_M(:)-XT(:))/norm(XT(:));
fprintf('Relative error = %0.8e\n',ErrorsM);
D=Xhat_M(:)-XT(:);
MSE_M = sum(D(:).*D(:))/numel(Xhat_M(:));
MAE_M=mean(mean(abs(D)));%平均绝对误差
fprintf('psnr_n1 = %0.2f  ssim_nl = %0.4f  fsim_nl = %0.4f MSE = %0.4f MAE = %0.4f ',psnr_M,ssim_M,fsim_M,MSE_M,MAE_M); 







