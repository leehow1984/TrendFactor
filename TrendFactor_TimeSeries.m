function [Result] = TrendFactor_TimeSeries(MarketData,MADays,LookBackFactor,reb_freq)

%/ Market Data: Financial Time Series Object
%/ MADays: Moving average day vector 1 X N
%/ SmoothingFactor: Moving Average beta smoothing vector 1 X N
%/ Rebalancing frequency: 'D' = Daily, 'W' = Weekly, 'M' = Monthly, 'Q'=
%/ Quaterly

%/ convert market data into the new rebalance frequency
NewfreqFTS = convertto(MarketData(max(MADays):end),reb_freq);
Dates = NewfreqFTS.dates;
Dates = Dates(LookBackFactor:end,:);
FtsInfo = ftsinfo(MarketData);


MADev_vec = zeros(0);
RetMat_t_vec = zeros(0);
b_vec = zeros(0);
stats_vec= zeros(0);
     
    % calculate time series of MA Deviation and Return 
    for i = 1:size(Dates,1)
        %/ get time series data for each security
        Security_tsObj = extfield(MarketData,FtsInfo.seriesnames{i,1});
        Security_Deviation = MA_Deviation_TimeSeries(Security_tsObj,MADays);
        %/ find deviation for all testing dates
        Security_Deviation = Security_Deviation(ismember(MarketData.dates,Dates,'rows'),:);
        
        Security_tsObj_freq = Security_tsObj(ismember(Security_tsObj.dates,Dates,'rows'));
        %/return
        RetMat_t =  fts2mat(Security_tsObj_freq(2:end))./ fts2mat(Security_tsObj_freq(1:end-1)) -1;
        

        
    end 
    
    for i = 1:size(MADev_vec,1)
        
        x = transpose(RetMat_t);
        y = [ones(size(transpose(MADev_tm1),1),1) transpose(MADev_tm1)];
        [b,bint,r,rint,stats] =  regress(x,y);
        b_tran = transpose(b);
        %/ beta coefficient vector
        b_vec = [b_vec; transpose(b)];
        stats_vec = [stats_vec; (stats)];
        
    end
    
    ExpRetRank_vec = zeros(0);
    RetMat_tplus1_vec = zeros(0);
    ExpRet_vec = zeros(0);
    quantile = 10;
    
    %/ predict return
    for i = LookBackFactor+1:size(Dates,1)-1
        %/calculate MA deviation at t 
        l_t = find(MarketData.dates == Dates(i-1)); 
        tsObj_t = MarketData(l_t-max(MADays)+1:l_t);
        MADev_t = MA_Deviation(tsObj_t,MADays); 
        %/ find smoothed beta coefficient
         b_smooth = mean(b_vec(i-LookBackFactor:i-1,:));
        %/ find expected return
         ExpRet = b_smooth(2:end) * MADev_t + b_smooth(1);
        %/ ExpRet = b_vec(i-1,2:end) * MADev_t + b_vec(i-1,1);
        %/ Ranking
        ExpRetRank = transpose(ceil(quantile * tiedrank(ExpRet) / length(ExpRet)));
        %/ actual future return
        l2_t = find(NewfreqFTS.dates == Dates(i)); 
        RetMat_tplus1 =  fts2mat(NewfreqFTS(l2_t))./ fts2mat(NewfreqFTS(l2_t-1)) -1;
        
        %/ export reslut
        ExpRet_vec = [ExpRet_vec; ExpRet];
        ExpRetRank_vec = [ExpRetRank_vec;transpose(ExpRetRank)];
        RetMat_tplus1_vec = [RetMat_tplus1;RetMat_tplus1_vec];
    end
    
    %/ analysis
    AverageRet = zeros(1,quantile);
    PandL_sub =  ones(1,quantile);
    PandL_tot = ones(1,1);
    for i = 1:size(ExpRetRank_vec,1)
        for j = 1:quantile
            AverageRet(1,j) = AverageRet(1,j) + mean(RetMat_tplus1_vec(i,ExpRetRank_vec(i,:)==j));
            PandL_sub(i+1,j) = PandL_sub(i,j) * (1 + mean(RetMat_tplus1_vec(i,ExpRetRank_vec(i,:)==j)));   
        end
           PandL_tot(i+1,1) = PandL_tot(i,1) * ( 1 + mean(RetMat_tplus1_vec(i,ExpRetRank_vec(i,:) == quantile)) - mean(RetMat_tplus1_vec(i,ExpRetRank_vec(i,:)==1)) ) ;
    end 
    
    
end 


