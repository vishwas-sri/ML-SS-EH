% Simulation Scenario
scenario.PU = [0 0]*1e3; 					% PU cartesian position in meters
scenario.Pr = 0.5; 							% PU transmission probability
scenario.TXPower = 0.1; 					% PU transmission power in mW
scenario.T = 5e-6; 							% SU spectrum sensing period in seconds
scenario.w = 5e6; 							% SU spectrum sensing bandwidth in hertz
scenario.NoisePSD_dBm = -153; 				% Noise PSD in dBm/Hz
scenario.NoisePower = (10^(scenario.NoisePSD_dBm/10)*1e-3)*scenario.w;


SuNumber=4;
scenario.SU = [zeros(1,SuNumber); linspace(0.5,1,SuNumber)]'*1e3; % SU cartesian position in meters
if (SuNumber==1)
    scenario.SU=[0 750];
end
scenario.fading = 'ray'; % Adds Rayleigh fading to the received signals
scenario.variance=2;
scenario.realiz = 50000; 						% MCS realization

trainingScenario = scenario;
trainingScenario.realiz = 250;



train = struct();
modelsHolder = struct();
epochs = 1;

%% Spectrum Sensing Procedure


[test.X, test.Y, ~,~,~,SNR] = MCS(scenario);
[train.X,train.Y,~,~,~,~] = MCS(trainingScenario);

meanSNR = mean(SNR(:,test.Y==1),2)
meanSNRdB = 10*log10(meanSNR)

csvwrite("Data/4_SNR.csv",meanSNR);
table=[train.X,train.Y];
csvwrite("Data/4_Train.csv",table);
table=[test.X,test.Y];
csvwrite("Data/4_Test.csv",table);
