load('E:\Dropbox\Dropbox\code\Research\research code\CSIMTL\python\save.mat');
a=reshape(outspa(1,1,:,:),[7,7]);
figure(1)
fmi=imagesc(a);
set(gca,'position',[0 0 1 1]); 
grid off;
axis normal;
axis off;
set(gca,'xtick',[]);
set(gca,'ytick',[]);