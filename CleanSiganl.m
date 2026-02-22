% load('USV.wav');
cd('C:\Users\owner\Desktop\Recording tests\Natans recordings')
[Example,Fs] = audioread('C57Bmom2.wav');
Example=double(Example);
Example=Example-mean(Example);
spectrogram(Example(:),5000,4500,5000,Fs,'yaxis')
%%
b=fir1(1500,[1 100]/500*2);
fe=filtfilt(b,1,Example);
fe = fe/max([max(fe) abs(min(fe))]) ;
audiowrite('USV_step1.wav' , fe , Fs)
%%
spectrogram(fe,5000,4500,5000,500,'yaxis')
axis([-Inf Inf 0 100]);
caxis([-60 -12])
%% Detect events
efe=fe.^2;
b=fir1(2500,0.1/50*2);
efe=filtfilt(b,1,efe);
%%
[p,inds]=findpeaks(efe); 
thr=prctile(p,86); % Selected by eyeballing the distribution
%%
tefe=efe;
clear pp
ii=1;
while max(tefe)>thr
    [m,ind]=max(tefe);
    st=min(find(tefe(1:ind)<thr/2,15000,'last'));
    en=max(find(tefe(ind+1:end)<thr/2,15000,'first'));
    pp(ii).loc=ind;
    pp(ii).size=m;
    pp(ii).st=st;
    pp(ii).en=en+ind;
    pp(ii).dur=ind-st+en;
    tefe(st:(ind+en))=thr/2;
    ii=ii+1;
end
%%
[~,I]=sort([pp.loc]);
pp=pp(I);
plot(efe);
% for ii=1:length(pp)
%     line(pp(ii).loc*[1 1],pp(ii).size*[0.9 1.1],'col','m');
%     line([pp(ii).st pp(ii).en],thr/2*[1 1],'col','k');
% end
lenthr=5000; %5 ms
lpp=pp([pp.dur]>lenthr);
for ii=1:length(lpp)
    line(lpp(ii).loc*[1 1],lpp(ii).size*[0.9 1.1],'col','r','linew',2);
    line([lpp(ii).st lpp(ii).en],thr/2*[1 1],'col','c','linew',4);
end
%% get call with guard regions (5 ms one each side)

allusvcalls = cell(size(lpp) + [1 0]);
%%
for ip =  1:length(lpp)
% ip = 1 ;
    call=fe((lpp(ip).st-500):(lpp(ip).en+500));
    [pxx,f]=pwelch(call,1000,500,5000,500); % resolution of 500 Hz (2 ms), by eyeballing
    [m,mind]=max(pxx);
    mfreq=f(mind);
    plot(f,10*log10(pxx));
    axis([0 100 -60 Inf])
    line(mfreq*[1 1],[-60 10*log10(m)],'col','r');
    [m,finds]=findpeaks(pxx,'MINPEAKDISTANCE',ceil(mind/6),...
        'MINPEAKHEIGHT',10^(-51.8/10),'SORTSTR','descend'); % Height threshold has to be fine-tuned
    tcall=call;
    nsig=zeros(size(call));
    ph=zeros(length(call),length(finds));
    partial=zeros(length(call),length(finds));
    amp=ph;

    for ii=1:min(length(finds),3)  %%% max 3 harmonies length(finds) % loop on spectral peaks
    %     etcall=filtfilt(b,1,tcall.^2);
    %     [~,st]=max(etcall);
        winlen=ceil(500/f(finds(ii))*7); % 7 cycles of the corresponding freq
        if mod(winlen,2)==0
            winlen=winlen+1;
        end
        win=dpss(winlen,2,1);
        win=win/max(win);
        winds=(-(winlen-1)/2):((winlen-1)/2);
        for ifor=(winlen+1)/2:length(tcall)-((winlen-1)/2) % loop in individual samples
            if mod(ifor,100)==0
                disp(ifor);
                disp([ii ip]);
            end
            toft=[tcall(ifor+winds).*win; zeros(length(win)*99,1)];
            ft=fft(toft);
            [~,mind]=max(abs(ft(600:800)));
            mind=mind+599;
            a=polyfit(-1:1,abs(ft(mind+[-1 0 1]))',2);
            mindmax=-a(2)/(2*a(1));
            fmindmax=mind+mindmax-1;
            c=exp(-1i*(2*pi/(100*winlen)*fmindmax*(0:(100*winlen-1))'));
            fc=sum(toft.*c);
            ph(ifor,ii)=angle(fc)+2*pi/(100*winlen)*fmindmax*((winlen+1)/2-1);
            amp(ifor,ii)=abs(fc);
        end
        partial(:,ii)=amp(:,ii).*cos(ph(:,ii))/winlen*4;
        plot([tcall(:) partial(:,ii)]);
%         pause;
        nsig=nsig+partial(:,ii);
        tcall=tcall-partial(:,ii);
    end
    allusvcalls{1,ip} = partial;
    allusvcalls{2,ip} = [m,finds];
%     ip = ip + 1 ;
end
% save allusvcalls
%%
clean_call = zeros(1,length(fe));
for i = 1:size(allusvcalls,2)
    ramp = ones ( 1 , size(allusvcalls{1,i},1)) ;
    ramp( 1 : 500 ) = (1 : 500)/500 ;
    ramp( (end-499) : end ) = (500 : -1 : 1 )/500 ;
    clean_call((lpp(i).st-500):(lpp(i).en+500)) = ramp'.*sum(allusvcalls{1,i},2) ;
end
    

    
%%
% sen = [];
% for i  = 1:size(allusvcalls,2)
%     sen = [sen (allusvcalls{2,i})'];
% end
        
    









