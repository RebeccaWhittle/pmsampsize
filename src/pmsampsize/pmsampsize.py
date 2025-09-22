import math 
import pandas as pd
from tabulate import tabulate, SEPARATING_LINE
import numpy as np
from scipy.stats import t, norm
import statsmodels.api as sm
import statsmodels.formula.api as smf


### binary
def pmsampsize_bin(rsquared,parameters,prevalence,shrinkage,cstatistic,noprint):
    r2a=rsquared
    n1=n2=n3=parameters

    if shrinkage < r2a:
        error_msg = f"User specified shrinkage is lower than R-squared adjusted. Error in log(1 - (r2a/shrinkage)) : nans produced"
        raise ValueError(error_msg)

# criteria 1 - shrinkage
    n1 = math.ceil(parameters / ((shrinkage - 1) * math.log(1 - (r2a / shrinkage))))
    shrinkage_1 = shrinkage
    E1 = n1*prevalence
    epp1 = E1/parameters
    EPP_1 = round(epp1, 2)

# criteria 2 - small absolute difference in r-sq adj
    lnLnull = (E1*(math.log(E1/n1)))+((n1-E1)*(math.log(1-(E1/n1))))
    max_r2a = (1 - math.exp((2*lnLnull)/n1))
    nag_r2 = r2a/max_r2a

    if max_r2a < r2a:
        error_msg = f"User specified R-squared adjusted is larger than the maximum possible R-squared ({max_r2a}) as defined by equation 23 (Riley et al. 2018)"
        raise ValueError(error_msg)

    s_4_small_diff = (r2a/(r2a+(0.05*max_r2a)))

    n2 = math.ceil((parameters/((s_4_small_diff-1)*(math.log(1-(r2a/s_4_small_diff))))))
    shrinkage_2 = s_4_small_diff

    E2 = n2*prevalence
    epp2 = E2/parameters
    EPP_2 = round(epp2, 2)

# criteria 3 - precise estimation of the intercept
    n3 = math.ceil((((1.96/0.05)**2)*(prevalence*(1-prevalence))))

    E3 = n3*prevalence
    epp3 = E3/parameters
    EPP_3 = round(epp3, 2)

    if shrinkage_2 > shrinkage:
        shrinkage_3 = shrinkage_2

    else:
        shrinkage_3 = shrinkage

# minimum n
    nfinal = max(n1,n2,n3)
    shrinkage_final = shrinkage_3
    E_final = nfinal*prevalence
    epp_final = E_final/parameters
    EPP_final = round(epp_final, 2)

# output table
    res = [["Criteria 1", n1, round(shrinkage_1, 3), parameters, rsquared, round(max_r2a, 3), round(nag_r2, 3), EPP_1], 
          ["Criteria 2", n2, round(shrinkage_2, 3), parameters, rsquared, round(max_r2a, 3), round(nag_r2, 3), EPP_2], 
          ["Criteria 3", n3, round(shrinkage_3, 3), parameters, rsquared, round(max_r2a, 3), round(nag_r2, 3), EPP_3],
           SEPARATING_LINE,
          ["Final SS", nfinal, round(shrinkage_final, 3), parameters, rsquared, round(max_r2a, 3), round(nag_r2, 3), EPP_final]]

    col_names = ["Criteria", "Sample size", "Shrinkage", "Parameter", "CS_Rsq", "Max_Rsq", "Nag_Rsq", "EPP"]

    if noprint==None:
        if cstatistic is not None:
            print("Given input C-statistic =", cstatistic, " & prevalence =", prevalence)
            print("Cox-Snell R-sq =", r2a, "\n",)
        print("NB: Assuming 0.05 acceptable difference in apparent & adjusted R-squared")
        print("NB: Assuming 0.05 margin of error in estimation of intercept")
        print("NB: Events per Predictor Parameter (EPP) assumes prevalence =", prevalence,"\n")
        print(tabulate(res, headers=col_names, numalign="right"))
        print(" \n", "Minimum sample size required for new model development based on user inputs = ", nfinal, "\n",
              "with ", math.ceil(E_final), " events (assuming an outcome prevalence = ", prevalence,") and an EPP = ", EPP_final, " \n", "\n", sep="")


    out = {
           "results_table": res,
           "final_shrinkage": shrinkage_final,
           "sample_size": nfinal,
           "parameters": parameters,
           "rsquared": r2a,
           "max_r2a": max_r2a,
           "nag_r2": nag_r2,
           "events": E_final,
           "EPP": EPP_final,
           "prevalence": prevalence,
           "type": "binary",
           "cstatistic": cstatistic
    }

    return out

### continuous
def pmsampsize_cont(rsquared,parameters,intercept,sd,shrinkage,mmoe,noprint):

    r2a = rsquared
    n = parameters + 2
    n1 = parameters + 2

# criteria 1
    es =  1 + ((parameters-2)/(n1*(math.log(1-((r2a*(n1-parameters-1))+parameters)/(n1-1)))))

    if es > shrinkage:
        shrinkage_1 = round(es, 3)
        n1 = n1
        spp_1 = n1/parameters
        SPP_1 = round(spp_1, 2)

    else: 
        while (es < shrinkage):
            n1 = n1 + 1
            es = 1 + ((parameters-2)/(n1*(math.log(1-((r2a*(n1-parameters-1))+parameters)/(n1-1)))))

            if not pd.isna(es) and es >= shrinkage:
                shrinkage_1 = round(es, 3)
                n1 = n1
                spp_1 = n1/parameters
                SPP_1 = round(spp_1, 2)

# criteria 2 - small absolute difference in r-sq adj & r-sq app
    n2 = math.ceil(1+((parameters*(1-r2a))/0.05))
    shrinkage_2 = 1 + ((parameters-2)/(n2*(math.log(1-((r2a*(n2-parameters-1))+parameters)/(n2-1)))))
    shrinkage_2 = round(shrinkage_2, 3)
    spp_2 = n2/parameters
    SPP_2 = round(spp_2, 2)

# criteria 3 - precise estimate of residual variance
    n3 = 234 + parameters

    shrinkage_3 = 1 + ((parameters-2)/(n3*(math.log(1-((r2a*(n3-parameters-1))+parameters)/(n3-1)))))
    shrinkage_3 = round(shrinkage_3, 3)
    spp_3 = n3 / parameters
    SPP_3 = round(spp_3, 2)

# criteria 4 - precise estimation of intercept
    n4 = max(n1,n2,n3)
    df = n4 - parameters - 1
    uci = intercept+((((sd**2*(1-r2a))/n4)**.5)*(-t.ppf(0.025, df)))
    lci = intercept-((((sd**2*(1-r2a))/n4)**.5)*(-t.ppf(0.025, df)))
    int_mmoe = uci / intercept

    if (int_mmoe > mmoe):
        while (int_mmoe > mmoe):
            n4 = n4 + 1
            df = n4 - parameters - 1
            uci = intercept + ((((sd**2*(1-r2a))/n4)**.5)*(t.ppf(0.025, df)))
            lci = intercept - ((((sd**2*(1-r2a))/n4)**.5)*(t.ppf(0.025, df)))
            int_mmoe = uci / intercept
        
            if (int_mmoe <= mmoe):
                shrinkage_4 = 1 + ((parameters-2)/(n4*(math.log(1-((r2a*(n4-parameters-1))+parameters)/(n4-1)))))
                shrinkage_4 = round(shrinkage_4, 3)
                spp_4 = n4 / parameters
                SPP_4 = round(spp_4, 2)
                int_uci = round(uci, 2)
                int_lci = round(lci, 2)
    else:
        shrinkage_4 = 1 + ((parameters-2)/(n4*(math.log(1-((r2a*(n4-parameters-1))+parameters)/(n4-1)))))
        shrinkage_4 = round(shrinkage_4, 3)
        spp_4 = n4 / parameters
        SPP_4 = round(spp_4, 2)
        int_uci = round(uci, 2)
        int_lci = round(lci, 2)

# minimum n
    nfinal = max(n1,n2,n3,n4)
    spp_final = nfinal / parameters
    SPP_final = round(spp_final, 2)
    shrinkage_final = 1 + ((parameters-2)/(nfinal*(math.log(1-((r2a*(nfinal-parameters-1))+parameters)/(nfinal-1)))))
    shrinkage_final = round(shrinkage_final, 3)

# output table
    res = [["Criteria 1", n1, round(shrinkage_1, 3), parameters, rsquared, SPP_1], 
          ["Criteria 2", n2, round(shrinkage_2, 3), parameters, rsquared, SPP_2], 
          ["Criteria 3", n3, round(shrinkage_3, 3), parameters, rsquared, SPP_3],
          ["Criteria 4", n4, round(shrinkage_4, 3), parameters, rsquared, SPP_4],
           SEPARATING_LINE,
          ["Final SS", nfinal, round(shrinkage_final, 3), parameters, rsquared, SPP_final]]

    col_names = ["Criteria", "Sample size", "Shrinkage", "Parameter", "Rsq", "SPP"]

    if noprint==None:
        print("\n","NB: Assuming 0.05 acceptable difference in apparent & adjusted R-squared","\n",
                  "NB: Assuming MMOE <= 1.1 in estimation of intercept & residual standard deviation","\n",
                  "SPP - Subjects per Predictor Parameter","\n","\n")       
        print(tabulate(res, headers=col_names, numalign="right"))
        print("\n","Minimum sample size required for new model development based on user inputs = ",nfinal,"\n","\n",
                  "* 95% CI for intercept = (",int_lci,", ",int_uci, "), for sample size n = ",nfinal, sep="") 

    out = {
          "results_table": res,
          "final_shrinkage": shrinkage_final,
          "sample_size": nfinal,
          "parameters": parameters,
          "r2a": r2a,
          "SPP": SPP_final,
          "int_mmoe": int_mmoe,
          "intercept": intercept,
          "int_uci": int_uci,
          "int_lci": int_lci,
          "type": "continuous"
    }

    return out


###survival
def pmsampsize_surv(rsquared,parameters,rate,timepoint,meanfup,shrinkage,mmoe,noprint):
    n=10000
    r2a = rsquared
    n1 = n2 = n3 = parameters
    tot_per_yrs = meanfup*n
    events = math.ceil(rate*tot_per_yrs)

    if (shrinkage < r2a):
        error_msg = f"User specified shrinkage is lower than R-squared adjusted. Error in log(1 - (r2a/shrinkage)) : nans produced"
        raise ValueError(error_msg)

# criteria 1 - shrinkage
    n1 = math.ceil((parameters/((shrinkage-1)*(math.log(1-(r2a/shrinkage))))))
    shrinkage_1 = shrinkage
    E1 = n1*rate*meanfup
    epp1 = E1/parameters
    EPP_1 = round(epp1, 2)

# criteria 2 - small absolute difference in r-sq adj
    lnLnull = (events*(math.log(events/n)))-events
    max_r2a = (1- math.exp((2*lnLnull)/n))
    nag_r2 = r2a/max_r2a

    if (max_r2a < r2a):
        error_msg = f"User specified R-squared adjusted is larger than the maximum possible R-squared ({max_r2a}) as defined by equation 23 (Riley et al. 2018)"
        raise ValueError(error_msg)

    s_4_small_diff = (r2a/(r2a+(0.05*max_r2a)))

    n2 = math.ceil((parameters/((s_4_small_diff-1)*(math.log(1-(r2a/s_4_small_diff))))))
    shrinkage_2 = s_4_small_diff

    E2 = n2*rate*meanfup
    epp2 = E2/parameters
    EPP_2 = round(epp2, 2)

# criteria 3 - precise estimation of the intercept
    n3 = max(n1,n2)
    tot_per_yrs = round(meanfup*n3, 1)
    uci = 1-(math.exp(-(rate+(1.96*((rate/(tot_per_yrs))**.5)))*timepoint))
    lci = 1-(math.exp(-(rate-(1.96*((rate/(tot_per_yrs))**.5)))*timepoint))
    cuminc = 1-(math.exp(timepoint*(rate*-1)))
    risk_mmoe = uci-cuminc

    n3 = n3
    E3 = n3*rate*meanfup
    epp3 = E3/parameters
    EPP_3 = round(epp3, 2)
    int_uci = round(uci, 3)
    int_lci = round(lci, 3)
    int_cuminc = round(cuminc, 3)

    if (shrinkage_2 > shrinkage):
        shrinkage_3 = shrinkage_2

    else:
        shrinkage_3 = shrinkage

# minimum n
    nfinal = max(n1,n2,n3)
    shrinkage_final = shrinkage_3
    E_final = nfinal*rate*meanfup
    epp_final = E_final/parameters
    EPP_final = round(epp_final, 2)
    tot_per_yrs_final = round(meanfup*nfinal,1)

# output table
    res = [["Criteria 1", n1, round(shrinkage_1, 3), parameters, rsquared, round(max_r2a, 3), round(nag_r2, 3), EPP_1], 
          ["Criteria 2", n2, round(shrinkage_2, 3), parameters, rsquared, round(max_r2a, 3), round(nag_r2, 3), EPP_2], 
          ["Criteria 3*", n3, round(shrinkage_3, 3), parameters, rsquared, round(max_r2a, 3), round(nag_r2, 3), EPP_3],
           SEPARATING_LINE,
          ["Final SS", nfinal, round(shrinkage_final, 3), parameters, rsquared, round(max_r2a, 3), round(nag_r2, 3), EPP_final]]

    col_names = ["Criteria", "Sample size", "Shrinkage", "Parameter", "CS_Rsq","Max_Rsq","Nag_Rsq", "EPP"]

    if noprint==None:
        print("\n","NB: Assuming 0.05 acceptable difference in apparent & adjusted R-squared","\n",
                  "NB: Assuming 0.05 margin of error in estimation of overall risk at time point =",timepoint,"\n",
                  "NB: Events per Predictor Parameter (EPP) assumes overall event rate =",rate,"\n","\n")          
        print(tabulate(res, headers=col_names, numalign="right"))
        print("\n","Minimum sample size required for new model development based on user inputs = ",nfinal,"\n",
              "corresponding to ",tot_per_yrs_final, " person-time** of follow-up, with ", math.ceil(E_final), " outcome events",
              "\n","assuming an overall event rate = ",rate, " and therefore an EPP = ",EPP_final,"\n",
              "* 95% CI for overall risk = (",int_lci,", ", int_uci, "), for true value of ", int_cuminc, " and sample size n = ",nfinal,
              "\n", "(**where time is in the units mean follow-up time was specified in)",sep="")

    out = {
           "results_table": res,
           "final_shrinkage": shrinkage_final,
           "sample_size": nfinal,
           "parameters": parameters,
           "r2a": r2a,
           "max_r2a": max_r2a,
           "nag_r2": nag_r2,
           "events": E_final,
           "EPP": EPP_final,
           "int_uci": int_uci,
           "int_lci": int_lci,
           "int_cuminc": int_cuminc,
           "rate": rate,
           "timepoint": timepoint,
           "meanfup": meanfup,
           "tot_per_yrs_final": tot_per_yrs_final,
           "type": "survival"
    }

    return out


###cstat2rsq
def cstat2rsq(cstatistic, prevalence, seed, noprint):

# define ss for simulation dataset
    n = 1000000

# define mu as a function of the C-statistic
    mu = math.sqrt(2) * norm.ppf(cstatistic)

# simulate large sample linear prediction based on two normals
# for non-eventsN(0, 1), events and N(mu, 1)
    np.random.seed(seed)
    LP = np.concatenate([
        np.random.normal(loc=0, scale=1, size=int(prevalence * n)),
        np.random.normal(loc=mu, scale=1, size=int((1 - prevalence) * n))
    ])
    y = np.concatenate([
        np.repeat(0, int(prevalence * n)),
        np.repeat(1, int((1 - prevalence) * n))
    ])

# Fit a logistic regression with LP as covariate;
# this is essentially a calibration model, and the intercept and
# slope estimate will ensure the outcome proportion is accounted
# for, without changing C-statistic
    data = {'y': y, 'LP': LP}
    model = smf.glm(formula='y ~ LP', data=data, family=sm.families.Binomial()).fit()

    R2_coxsnell = 1 - np.exp(-1 * (model.null_deviance - model.deviance) / n)
    result = {
           "R2_coxsnell": R2_coxsnell
    }
    return result

### error check
def pmsampsize_errorcheck(type,csrsquared,nagrsquared,rsquared,parameters,shrinkage,cstatistic,prevalence,rate,timepoint,meanfup,intercept,sd,mmoe,noprint):

# syntax checks
    if type not in ["c", "b", "s"]:
        raise ValueError('type must be "c" for continuous, "b" for binary, or "s" for survival')
    if not isinstance(parameters, int):
            raise ValueError('parameters must be an integer')
    if parameters != round(parameters):
            raise ValueError('parameters must be an integer')

# parameter restrictions
    if shrinkage <= 0 or shrinkage >= 1:
        raise ValueError('shrinkage must be between 0 and 1')

#### parameters for continuous
    if type == "c":
 # syntax
        if rsquared is None:
            raise ValueError('rsquared must be specified for continuous outcome models')
 # parameters needed
        if sd is None:
            raise ValueError('sd must be specified for continuous sample size')
        if intercept is None:
            raise ValueError('intercept must be specified for continuous sample size')
 # parameter conditions
        if not isinstance(rsquared, (int, float)):
            raise ValueError('rsquared must be numeric')
        if not isinstance(intercept, (int, float)):
            raise ValueError('intercept must be numeric')
        if not isinstance(sd, (int, float)):
            raise ValueError('sd must be numeric')     
 # parameters not needed
        if prevalence is not None:
            raise ValueError('prevalence not required for continuous sample size')
        if rate is not None:
            raise ValueError('rate not required for continuous sample size')
        if timepoint is not None:
            raise ValueError('timepoint not required for continuous sample size')
        if meanfup is not None:
            raise ValueError('meanfup not required for continuous sample size')

#### parameters for binary
    if type == "b":
 # parameters needed
        if prevalence is None:
            raise ValueError('prevalence must be specified for binary sample size')
 # parameter conditions
        if not isinstance(prevalence, (int, float)):
            raise ValueError('prevalence must be numeric')

        if cstatistic is not None:
            if cstatistic < 0 or cstatistic > 1:
                raise ValueError('cstatistic must be between 0 and 1')
            if not isinstance(cstatistic, (int, float)):
                raise ValueError('cstatistic must be numeric')

        if nagrsquared is not None:
            if nagrsquared < 0 or nagrsquared > 1:
                raise ValueError('nagrsquared must be between 0 and 1')
            if not isinstance(nagrsquared, (int, float)):
                raise ValueError('nagrsquared must be numeric')

        if csrsquared is not None:
            if csrsquared < 0 or csrsquared > 1:
                raise ValueError('csrsquared must be between 0 and 1')
            if not isinstance(csrsquared, (int, float)):
                raise ValueError('csrsquared must be numeric')
            
        if prevalence <= 0 or prevalence >= 1:
            raise ValueError('prevalence must be between 0 and 1')
    
 # parameters not needed
        if rate is not None:
            raise ValueError('rate not required for binary sample size')
        if timepoint is not None:
            raise ValueError('timepoint not required for binary sample size')
        if meanfup is not None:
            raise ValueError('meanfup not required for binary sample size')
        if intercept is not None:
            raise ValueError('intercept not required for binary sample size')
        if sd is not None:
            raise ValueError('sd not required for binary sample size')
 
#### parameters for survival
    if type == "s":
 # parameters needed
        if rate is None:
            raise ValueError('rate must be specified for survival sample size')
        if timepoint is None:
            raise ValueError('timepoint must be specified for survival sample size')
        if meanfup is None:
            raise ValueError('meanfup must be specified for survival sample size')

 # parameter conditions
        if nagrsquared is not None:
            if nagrsquared < 0 or nagrsquared > 1:
                raise ValueError('nagrsquared must be between 0 and 1')
            if not isinstance(nagrsquared, (int, float)):
                raise ValueError('nagrsquared must be numeric')
        if csrsquared is not None:
            if csrsquared < 0 or csrsquared > 1:
                raise ValueError('csrsquared must be between 0 and 1')
            if not isinstance(csrsquared, (int, float)):
                raise ValueError('csrsquared must be numeric')

        if not isinstance(rate, (int, float)):
            raise ValueError('rate must be numeric')
        if not isinstance(timepoint, (int, float)):
            raise ValueError('timepoint must be numeric')
        if not isinstance(meanfup, (int, float)):
            raise ValueError('meanfup must be numeric')

        if rate <= 0:
            raise ValueError('the specified overall event rate must be greater than 0')
        if timepoint <= 0:
            raise ValueError('the timepoint of interest for prediction must be greater than 0')
        if meanfup <= 0:
            raise ValueError('the average mean follow-up time must be greater than 0')
 # parameters not needed
        if prevalence is not None:
            raise ValueError('prevalence not required for survival sample size')
        if intercept is not None:
            raise ValueError('intercept not required for survival sample size')
        if sd is not None:
            raise ValueError('sd not required for survival sample size')
        

### wrapper
def pmsampsize(type, parameters, nagrsquared = None, csrsquared = None, rsquared = None,
               shrinkage = 0.9,prevalence = None, cstatistic = None,seed = 123456,rate = None,
               timepoint = None,meanfup = None,intercept = None,sd = None,mmoe=1.1,noprint=None):
    
    """Computes the minimum sample size required for the development of a new multivariable prediction model
 
    Parameters
    ----------
    type: str
        specifies the type of analysis for which sample size is being calculated
        "b" specifies sample size calculation for a prediction model with a binary outcome
        "c" specifies sample size calculation for a prediction model with a continuous outcome
        "s" specifies sample size calculation for a prediction model with a survival (time-to-event) outcome
    nagrsquared: float
        for type="b" or type="s" this specifies the expected value of the Nagelkerke's R-squared of the new model, 
        which is the Cox-Snell R-squared scaled to lie in the [0,1] range. 
        It is interpretable in the same way as the standard R-squared, i.e. the percentage of variation in outcome values explained by the model. 
        Please read the description of rsquared for additional details about specifying the expected R-squared performance
    csrsquared: float
        for type="b" or type="s" this specifies the expected value of the Cox-Snell Rsquared of the new model. 
        The Cox-Snell R-squared is the generalised version of the well-known R-squared for continuous outcomes, based on the likelihood. 
        Please read the description of rsquared for additional details about specifying the expected R-squared performance. 
        The papers by Riley et al. (see references) outline how to obtain the Cox-Snell R-squared value from published studies if they are not reported, 
        using other information (such as the C-statistic [see cstatistic() option below]).
    rsquared: float
        for type="c" this specifies the expected value of the R-squared of the new model, where R-squared is the percentage of variation in outcome values explained by the model. 
        For example, the user may input the value of the Rsquared reported for a previous prediction model study in the same field. 
        If taking a value from a previous prediction model development study, users should input the model's adjusted R-squared value, 
        not the apparent R-squared value, as the latter is optimistic (biased). 
        However, if taking the R-squared value from an external validation of a previous model, the apparent R-squared can be used 
        (as the validation data was not used for development, and so R-squared apparent is then unbiased). 
        Users should be conservative with their chosen R-squared value; for example, by taking the R-squared value from a previous model, 
        even if they hope their new model will improve performance.
    parameters: int
        specifies the number of candidate predictor parameters for potential inclusion in the new prediction model. 
        Note that this may be larger than the number of candidate predictors, as categorical and continuous predictors often require two or more parameters to be estimated.
    shrinkage: float, default=0.9
        specifies the level of shrinkage desired at internal validation after developing the new model. 
        Shrinkage is a measure of overfitting, and can range from 0 to 1, with higher values denoting less overfitting. 
        We recommend a shrinkage = 0.9 (the default in pmsampsize), which indicates that the predictor effect (beta coefficients) 
        in the model would need to be shrunk by 10% to adjust for overfitting. See references below for further information
    prevalence: float
        (type="b" option) specifies the overall outcome proportion (for a prognostic model) or overall prevalence (for a diagnostic model) expected within the model development dataset. 
        This should be derived based on previous studies in the same population.
    cstatistic: float
        (type="b" option) specifies the C-statistic reported in an existing prediction model study to be used in conjunction with the expected prevalence 
        to approximate the Cox-Snell R-squared using the approach of Riley et al. 2020. Ideally, this should be an optimism-adjusted C-statistic. 
        The approximate Cox-Snell Rsquared value is used as described above for the csrsquared() option, 
        and so is treated as a baseline for the expected performance of the new model.
    seed: int, default=123456
        (type="b" option) specifies the initial value of the random-number seed used by the random-number functions 
        when simulating data to approximate the CoxSnell R-squared based on reported C-statistic 
        and expected prevalence as described by Riley et al. 2020
    rate: float
        (type="s" option) specifies the overall event rate in the population of interest, 
        for example as obtained from a previous study, for the survival outcome of interest. 
        NB: rate must be given in time units used for meanfup and timepoint options.
    timepoint: float
        (type="s" option) specifies the timepoint of interest for prediction. 
        NB: time units must be the same as given for meanfup option (e.g. years, months).
    meanfup: float
        (type="s" option) specifies the average (mean) follow-up time anticipated for individuals in the model development dataset, 
        for example as taken from a previous study in the population of interest. NB: time units must be the same as given for timepoint option.
    intercept: float
        (type="c" options) specifies the average outcome value in the population of interest 
        e.g. the average blood pressure, or average pain score. 
        This could be based on a previous study, or on clinical knowledge.
    sd: float
        (type="c" options) specifies the standard deviation (SD) of outcome values in the population 
        e.g. the SD for blood pressure in patients with all other predictors set to the average. 
        This could again be based on a previous study, or on clinical knowledge.
    mmoe: float, default=1.1
        (type="c" options) multiplicative margin of error (MMOE) acceptable for calculation of the intercept. 
        The default is a MMOE of 10%. 
        Confidence interval for the intercept will be displayed in the output for reference. 
        See references below for further information.
    noprint: bool, default=False
        supresses output being printed                                                
    """   

  # error checking
    pmsampsize_errorcheck(type=type,nagrsquared=nagrsquared,csrsquared=csrsquared,rsquared=rsquared,parameters=parameters,shrinkage=shrinkage,cstatistic=cstatistic,
                          prevalence=prevalence,rate=rate,timepoint=timepoint,meanfup=meanfup,intercept=intercept,sd=sd,mmoe=mmoe,noprint=noprint)  

  # choose function based on analysis type
    if type == "c":
        out = pmsampsize_cont(rsquared=rsquared,parameters=parameters,intercept=intercept,
                                            sd=sd,shrinkage=shrinkage,mmoe=mmoe,noprint=noprint)


    if type == "b":
        rsquared = None

        if nagrsquared is not None:
            if csrsquared is not None:
                raise ValueError("Only one of csrsquared() or nagrsquared() can be specified")
            E = parameters*prevalence
            lnLnull = (E*(math.log(E/parameters)))+((parameters-E)*(math.log(1-(E/parameters))))
            max_r2a = (1- math.exp((2*lnLnull)/parameters))
            rsquared = round(nagrsquared*max_r2a,3)

        if csrsquared is not None:
            if nagrsquared is not None:
                raise ValueError("Only one of csrsquared() or nagrsquared() can be specified")
            rsquared = csrsquared

        if cstatistic is not None:
            if rsquared is not None:
                raise ValueError("Only one of csrsquared() or nagrsquared() or cstatistic() can be specified")

            approx_rsq = cstat2rsq(cstatistic=cstatistic, prevalence=prevalence, seed=seed, noprint=noprint)
            rsquared = round(approx_rsq["R2_coxsnell"], 4)

        else: 
            if rsquared is None:
                raise ValueError("One of csrsquared() or nagrsquared() or cstatistic() must be specified")

        out = pmsampsize_bin(rsquared=rsquared,parameters=parameters,prevalence=prevalence,
                                         shrinkage=shrinkage,cstatistic=cstatistic,noprint=noprint)

    if type == "s":
        rsquared = None        
        if nagrsquared is not None:
            if csrsquared is not None:
                raise ValueError("Only one of csrsquared() or nagrsquared() can be specified")

            events = parameters*rate*meanfup
            lnLnull = (events*(math.log(events/parameters)))-events
            max_r2a = (1- exp((2*lnLnull)/parameters))
            rsquared = round(nagrsquared*max_r2a,3)

        if csrsquared is not None:
            if nagrsquared is not None:
                raise ValueError("Only one of csrsquared() or nagrsquared() can be specified")
            rsquared = csrsquared    

        else: 
            if rsquared is None:
                raise ValueError("One of csrsquared() or nagrsquared() or cstatistic() must be specified")

        out = pmsampsize_surv(rsquared=rsquared,parameters=parameters,rate=rate,
                                          timepoint=timepoint,meanfup=meanfup,shrinkage=shrinkage,mmoe=mmoe,noprint=noprint)
    
    return out