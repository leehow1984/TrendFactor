function [Deviation] = MA_Deviation_TimeSeries(tsObj,MADays)

%/ calculate MA Deviation for one single security


    Deviation = zeros(0);
    
    TS = fts2mat(tsObj); %/ export data
    for i = 1:size(MADays,2)
        MA = tsmovavg(TS,'s',MADays(1,i),1);
        Deviation = [Deviation,MA./TS]; %/ calculate divation from MA
    end

end
