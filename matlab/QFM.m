%CSI visualization: visualize the data of each TR Pair, open untitled1.fig first, then run this code

filepath='H:\Widar3.0\CSI1\20181209\user6\'; %raw CSI path
sapath='H:\Widar3.0\QFM\FME\';
uname='user6';%user name
suname='15';%save file name

for mn=1:6
    for ln=1:5
        for on=1:5
            for rn=1:5 
                for rsn=1:6
                    mfm=zeros(1,3);
                    filename=[filepath,uname,'-',num2str(mn),'-',num2str(ln),'-',num2str(on),'-',num2str(rn),'-r',num2str(rsn),'.dat'];
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
                                 package((j2-1)*30+row,i-k)=csi(j1,j2,j3);
                            end
                        end
					   package1=abs(package((j2-1)*30+1:(j2-1)*30+30,:));
                       mf=mean(mean(package1)./var(package1));
                       mfm(1,j2)=mf;
                    end
                    [~,nma]=max(mfm);
                    [~,nmi]=min(mfm);
                    csiqdata=package((nma-1)*30+1:(nma-1)*30+30,:)./package((nmi-1)*30+1:(nmi-1)*30+30,:);%CSI ratio
                    qfm(:,:)=angle(csiqdata(:,:));%obtain phase
                    fmi=imagesc(qfm);%matrix to image
                    set(gca,'position',[0 0 1 1]); 
                    grid off;
                    axis normal;
                    axis off;
                    set(gca,'xtick',[]);
                    set(gca,'ytick',[]);
                    sname=[suname, '-' ,num2str(mn),'-',num2str(ln),'-',num2str(on),'-',num2str(rn),'-',num2str(rsn)];
                    saveas(fmi,strcat(sapath,sname,'.jpg'));
                    disp(['save',sname,'success.']);
                end
            end
        end
    end
end