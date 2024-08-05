%Combine the image of each TR pair 

filepath='H:\Widar3.0\QFM\FME\'; %image file path of each TR pair
sapath='H:\Widar3.0\QFM\STIFMM\';
uname='15';

for mn=1:6
    for ln=1:5
        for on=1:5
            for rn=1:5
                filename=[filepath,uname,'-',num2str(mn),'-',num2str(ln),'-',num2str(on),'-',num2str(rn),'-',num2str(1),'.jpg'];
                fm=imread(filename);
                sfm=fm;
                for rsn=2:6
                    filename=[filepath,uname,'-',num2str(mn),'-',num2str(ln),'-',num2str(on),'-',num2str(rn),'-',num2str(rsn),'.jpg'];
                    fm=imread(filename);
                    sfm=[sfm;fm];
                end
                sname=[uname, '-' ,num2str(mn),'-',num2str(ln),'-',num2str(on),'-',num2str(rn),'.jpg'];
                sfm=imresize(sfm, 0.28);
                imshow(sfm);
                imwrite(sfm, strcat(sapath,sname));
                disp(['save',sname,'success.']);
            end
        end
    end
end