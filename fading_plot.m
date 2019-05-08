%% Define codebooks
close all;
clear;

% Create Data
index=(0:15).';
data=de2bi(index);

% (7,4) Hamming code
G_Hamming=[1 0 0 0 1 1 0;
    0 1 0 0 1 0 1;
    0 0 1 0 0 1 1;
    0 0 0 1 1 1 1];
Ham_cb=mod(data*G_Hamming,2);

% (4,4) Uncoded
G_uncoded=[1 0 0 0;
           0 1 0 0;
           0 0 1 0;
           0 0 0 1];
uncoded_cb=mod(data*G_uncoded,2);

load('channel_test.mat', 'h_test');
h_test=h_test(:,1:7);

%% Simulation
% Call DNN scheme BER
% 
load('DNN.mat')

%EbNodB=-4:8;
Numsamp=size(h_test,1);
EbNo=10.^(EbNodB./10);
rate=4/7;

% EbNo to SNR conversion
SNR=2*rate*EbNo; % SNRdB= 10xlog_10 (SNR)

% Store Number of errors
blockerror_Ham=zeros(1,length(SNR));
blockerror_uncoded=zeros(1,length(SNR));

ind=0;
for ebno=EbNo
    ind=ind+1;
    fprintf('EbNo_dB = %d\n',EbNodB(ind));
    err_Ham=0;
    err_uncoded=0;
    for ii=1:Numsamp
        data_ind=randi([0,15]);
        dd=de2bi(data_ind,4);
        
        
        cw_Ham=mod(dd*G_Hamming,2);
        % Modulate Hamming codeword
        x_Ham=cw_Ham*2-1;
        x_uncoded=dd*2-1;
        % Channel
        y_Ham=h_test(ii,:).*x_Ham+randn(1,7)/sqrt(2*rate*ebno);
        y_uncoded=h_test(ii,1:4).*x_uncoded+randn(1,4)/sqrt(2*ebno);
        snr_Ham=h_test(ii,:).^2.*(2*rate*ebno);
        snr_uncoded=h_test(ii,1:4).^2.*(2*ebno);
        mmse_Ham=snr_Ham./(snr_Ham+1);
        mmse_uncoded=snr_uncoded./(snr_uncoded+1);
        [dec_Ham_cw, dec_Ham_ind]=mldec_fading(y_Ham,Ham_cb, h_test(ii,:));
        [dec_uncoded_cw, dec_uncoded_ind]=mldec_fading(y_uncoded,uncoded_cb, h_test(ii,1:4));
        %err=err+sum(cw~=dec);
        if dec_Ham_ind~=(data_ind+1)
            err_Ham=err_Ham+1;
        end
        if dec_uncoded_ind~=(data_ind+1)
            err_uncoded=err_uncoded+1;
        end
    end
    blockerror_Ham(ind)=err_Ham./Numsamp;
    blockerror_uncoded(ind)=err_uncoded./Numsamp;
end

figure;
semilogy(EbNodB, blockerror_Ham, 'b--', 'LineWidth',2);
hold on;
semilogy(EbNodB, blockerror_uncoded, 'k', 'LineWidth',2);

set(gca,'FontSize',16)
xlabel('EbNo [dB]');
ylabel('Block Error Rate');
grid on;

semilogy(EbNodB, ber, 'r', 'LineWidth',2)

legend('(7,4) Hamming ML', 'Uncoded (4,4)', 'DNN (7,4)')

save result.mat
