filepath='H:\Widar3.0\CSI1\20181130_user5_10_11\user10\user10-3-1-3-1'; %路径
sapath='E:\Dropbox\Dropbox\file\Research\Dual attention based gesture recognition\figure\zs\';
uname='user6';
suname='0';


% for rsn=1:6
    mfm=zeros(1,3);
    filename=[filepath,'-r',num2str(3),'.dat'];
    c1 = read_bf_file(filename);
    dl=length(c1);
    qfm=zeros(30,dl);
    dt=1;
    k=0;
    Num_subcarrier=30;
    package=zeros(90,dl);
    csi_trace=c1(dt:dl,1);
    for j2=1:3
        for i=1:length(c1)
            row=0;
            csi_entry = csi_trace{i};
            csi=get_scaled_csi(csi_entry);
            j1=1; 
            for j3=1:Num_subcarrier
                 row=row+1;
                 package((j2-1)*30+row,i-k)=csi(j1,j2,j3); %获取CSI值
            end
        end
       package1=abs(package((j2-1)*30+1:(j2-1)*30+30,:)); %获取CSI幅度值
       mf=mean(mean(package1)./var(package1));
       mfm(1,j2)=mf;
    end
    [~,nma]=max(mfm);
    [~,nmi]=min(mfm);
    csiqdata=package((nma-1)*30+1:(nma-1)*30+30,:)./package((nmi-1)*30+1:(nmi-1)*30+30,:);
    qfm(:,:)=angle(csiqdata(:,:));
    figure(1);
    fmi=plot(qfm(13,:));
%     set(gca,'position',[0 0 1 1]); 
%     grid off;
%     axis normal;
%     axis off;
%     set(gca,'xtick',[]);
%     set(gca,'ytick',[]);
    sname=[suname,'-',num2str(rsn)];
    saveas(fmi,strcat(sapath,sname,'.jpg'));
    %disp(['save',sname,'success.']);
% end
