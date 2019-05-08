%% Define codebooks
close all;
clear;

load('channel_test.mat', 'h_test');
h_test=h_test(:,1:2);
rate=2;

%% Simulation
% Call DNN scheme BER
% 
load('DNN.mat')

%EbNodB=-4:8;
Numsamp=size(h_test,1);
EbNo=10.^(EbNodB./10);


% EbNo to SNR conversion
SNR=2*rate*EbNo; % SNRdB= 10xlog_10 (SNR)

ber_qam=zeros(1,length(SNR));
ber_psk=zeros(1,length(SNR));

ind=0;
for ebno=EbNo
    ind=ind+1
    ebno_in=ebno/2 * sum(h_test.^2./2,2);
    [~, ser_psk]= berawgn(ebno_in(1:1000), 'psk', 16, 'nondiff');
    [~, ser_qam]= berawgn(ebno_in, 'qam', 16);
    ber_qam(ind) = mean(ser_qam,1);
    ber_psk(ind) = mean(ser_psk,1);
    %3*mean(qfunc(sqrt(4/5*sum(h_test.^2./2,2).*EbNo(ind)/2)),1);
end


semilogy(EbNodB, ber, 'r', 'LineWidth',2)
hold on;
semilogy(EbNodB, ber_psk, 'k', 'LineWidth',2)
semilogy(EbNodB, ber_qam, 'b', 'LineWidth',2)



set(gca,'FontSize',16)
xlabel('EbNo [dB]');
ylabel('Block Error Rate');
grid on;


legend('DNN (2,4)', '16-PSK', '16-QAM')

%save result.mat
