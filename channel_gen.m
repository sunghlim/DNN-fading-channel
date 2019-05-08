function channel_gen(N_train, N_test, code_length, sigma)

if nargin==3
    h_train=randn(N_train, code_length);
    %h_train(1:fix(N_train/8), :)= ones(fix(N_train/8), code_length);
    ind=randperm(N_train);
    N_AWGN=fix(N_train/10);
    h_train(ind(1:N_AWGN),:) = ones(N_AWGN, code_length);
    %h_train = ones(N_train, code_length);
    h_test=randn(N_test, code_length);
    save('channel_train.mat', 'h_train');
    save('channel_test.mat', 'h_test');
elseif nargin==3
    
    h_train=randn(N_train, code_length);
    h_test=randn(N_test, code_length);
    h_train_est = h_train + sigma*randn(N_train, code_length);
    h_test_est = h_train + sigma*randn(N_test, code_length);

    save('channel_train.mat', 'h_train', 'h_train_est');
    save('channel_test.mat', 'h_test','h_test_est');
end