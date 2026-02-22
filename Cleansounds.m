%% cleaning sounds
% The algorithem is:
% 1. Converting to wav file
% 1.a - for this it need to be normalized
% 2. Using OMLSA
% 3. Saving to mat file

%% Set Variables
% Fs = 500000; % usualy it is 500000
% fn = 'C:\Users\owner\Desktop\Recording tests\Natans recordings';
% fn = [fn 'USV'];

[file,folder] = uigetfile('*.wav' , 'Choose a wav file to clean') ;
fn = [folder,file] ;
cd(folder)
[data,Fs] = audioread(file);

% [data,si,h]=abfload(filen,'start', 0, 'stop', 'e');

%% Load the sound mat file
% load(fn)
fn = fn(1:end-4);
% save to wav
norm = data(:,1)./max(abs(data(:,1)));
audiowrite([fn '.wav'],norm,Fs);
fnOut = [fn '_clean_new'];
%% call OMLSA
[in,out] = omlsa(fn,fnOut);

%% viewing output and saving mat file
[file,folder] = uigetfile('*.wav' , 'Choose a wav file to clean') ;
[y,Fs1] = audioread(file);
norm = norm(1:length(y)) ;
figure;
hold all
plot(norm)
noise = norm - y;
% plot(noise);
plot(y);
save(fnOut,'y');

%%
winlen = 512;
win=dpss(winlen,2,1);
win=win/max(win);
h = figure;
subplot(3,1,1)
spectrogram(y,win,winlen/2,winlen*4,Fs,'yaxis');
caxis([-80 -40])
ylim([0 90000]);
subplot(3,1,2)
spectrogram(double(norm),win,winlen/2,winlen*4,Fs,'yaxis');
caxis([-80 -40])
ylim([0 90000]);
subplot(3,1,3)
spectrogram(double(noise),win,winlen/2,winlen*4,Fs,'yaxis');
caxis([-80 -40])
ylim([0 90000]);



