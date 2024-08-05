tic;
na='user10-3-1-3-1-r6'; %文件名
filepath='H:\Widar3.0\CSI1\20181130_user5_10_11\user10\'; %路径
emotion='angry/';
path=[filepath, na, '.dat'];
c1 = read_bf_file(path);
dl=length(c1);
dt=1;
Num_subcarrier=30;
csi_trace=c1(dt:dl,1);
package=zeros(30,dl);
k=0;
y=[];
for i=1:length(c1)
  row=0;
  csi_entry = csi_trace{i};
  csi=get_scaled_csi(csi_entry);
%   [h,~,~]=size(csi);
%   if h<2
%       k=k+1;
%       continue;
%   end
 j1=1; %选择发送天线1th
 j2=2; %选择接收天线3th
     for j3=1:Num_subcarrier
          row=row+1;
          package(row,i-k)=csi(j1,j2,j3); %获取CSI值
     end
end
dl=dl-k;
package1=abs(package); %获取CSI幅度值
%mean(mean(package1)./var(package1))
plot(abs(package(1,:)));
% package=package(2,1:dl); %选择第二个子载波
% %%巴特沃斯低通滤波器设计
% sa = 1000;      fn = sa/2;      % Sampling frequency (1000Hz), Nyquist frequency(500Hz)
% fp = 10;        fs =30;       % Passband (0~40Hz), Stop band(150Hz-500Hz)
% Wp = fp/fn;     Ws = fs/fn;    % Normalized passband (40/500), stopband (150/500)
% Rp = 3;         Rs = 60;        % ripple less than 3 dB, attenuation larger than 60dB
% [n,Wn] = buttord(Wp,Ws,Rp,Rs);  % Returns n = 5; Wn=0.0810; 
% [b,a] = butter(n,Wn);           % Designde signs an order n lowpass digital
% freqz(b,a,512,sa);              % returns the frequency response vector h and 
% title('n=5 Butterworth Lowpass Filter');
% % for j=1:30
% for i=1:length(package(1,1:dl))
%     if isinf(package(1,i))
%                 package(1,i)=package(1,i-1);
%     end
% end
% %% 滤波与绘图
% xn=abs(package(1,1:dl));
% y= filter(b, a, xn);          % Implement designed filter
% plot(y);
% hold on;
% % end
% figure(1);
% subplot(2,1,1);
% t=1:dl;
% plot(t, xn);
% title('Before Filter');
% subplot(2,1,2);
% figure(2);
%  plot(y);
% title('After Filter');
toc;