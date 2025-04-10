import re
import math

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pandas.plotting import scatter_matrix

# TODO -> Lint code (add params types and etc) 
# TODO -> utils file with smoothing and other functions in script
# TODO -> Load constant values from yaml or json
# Total carbon ----------------------------------------
soc_total = 3.08e9

# Respiration Rate Parameters ----------------------------------------
k_soc = 0.2    # Default value, change to 0.5 for wetlands younger than 3 years
kM_lab = 0.0018 # μmol/m3 Half-saturation constant for labile carbon 
kM_soc_avail = 0.0018 # μmol/m3 Half-saturation constant for avail carbon 
alpha_soc = 2.2e4  # μmolCO2/s Pre-exponential factor for SOC 
alpha_lab = 3e2  # μmolCO2/s Pre-exponential factor for labile carbon 
EA_soc = 21e3 # J/mol Activation energy  for soc
EA_lab = 17.2e3 # J/mol Activation energy  for labile

# GPP Parameters ----------------------------------------
# LUE = 0.56 # g biomass/MJ PAR Light Use Efficiency, example value 
# PAR = 4.272e9 # MJ/m2.s Photosynthetically active radiation 
k = 0.2 # Light attenuation coefficient
Ha = 450 # (J/mol) Activation energy for photosynthesis increase 
Hd = 350 # (J/mol) Deactivation energy for photosynthesis decrease 
T_opt = 298 # Optimum temperature in Kelvin (25°C)

# TODO -> Static functions ----------------------------------------
def LAI_smooth_func(DOY):
    ...
# LAI_smooth_func <- function(DOY) {
#   doy_clamped <- pmax(min(LAI$DOY), pmin(max(LAI$DOY), DOY))
#   predict(LAI_smooth_interpolation, newdata = data.frame(DOY = doy_clamped))
# }

def LUE_smooth_func(DOY):
    ...
# LUE_smooth_func <- function(DOY) {
#   doy_clamped <- pmax(min(LUE$DOY), pmin(max(LUE$DOY), DOY))
#   predict(LUE_smooth_interpolation, newdata = data.frame(DOY = doy_clamped))
# }

def f_WT_smooth_func(DOY):
    ...
# f_WT_smooth_func <- function(DOY) {
#   predict(f_WT_smooth_interpolation, newdata = data.frame(DOY = DOY))
# }

def modCost():
    ...

def modFit():
    ...

def sensFun():
    ...

def collin():
    ...


def cost_calculation(fit_pars, data, response, params1):
    fixpars = {k: v for k, v in params1.items() if k not in fit_pars}
    pars = pd.Series(fixpars.values() + fit_pars)
    
    out = ghg_model(
        DOY = data["DOY"],
        Temp = data["Temp"],
        PAR = data["PAR"],
        params = pars
    )
    obs = data.frame(DOY = data["DOY"], NEE = data[[response]])
    
    return(modCost(model = out, obs = obs, x = "DOY"))


# LAI ----------------------------------------
def process_LAI():
    "Function to process LAI"
    LAI = pd.read_excel("data/OL1/LAI.xlsx")
    LAI["datetime"] = LAI["raster"].apply(lambda x : pd.to_datetime(x[7:17]))
    LAI = LAI.rename(columns={"raster_mean": "LAI"})
    LAI["DOY"] = LAI["datetime"].dt.dayofyear
    # True if DOY or LAI are NaN
    nan_idxs = LAI[["DOY", "LAI"]].isna().any(axis=1)
    LAI = LAI.drop(np.where(nan_idxs == True)[0])
    LAI = LAI.reset_index(drop=True)

    # Linear interpolation of LAI Data
    lowess = sm.nonparametric.lowess(LAI["LAI"], LAI["DOY"], frac=0.5)  # frac is the smoothing parameter
    LAI_smooth_interpolation = pd.DataFrame(lowess, columns=['DOY', 'LAI_smoothed'])

    return LAI_smooth_interpolation

# LUE ----------------------------------------
def process_LUE():
    "Function to process LUE"
    LUE = pd.read_excel("data/OL1/CIgreen.xlsx")
    regex = r"(\d{4}-\d{2}-\d{2})"
    LUE["Date"] = LUE["raster"].apply(lambda x: pd.to_datetime(re.search(regex, x).group(0)))
    LUE["DOY"] = LUE["Date"].dt.dayofyear
    LUE["LUE"] = LUE["raster_mean"].apply(lambda x: (0.04*x) + 0.001)

    # Linear interpolation of LUE data 
    lowess = sm.nonparametric.lowess(LUE["LUE"], LUE["DOY"], frac=0.5)  # frac is the smoothing parameter
    LUE_smooth_interpolation = pd.DataFrame(lowess, columns=['DOY', 'LUE_smoothed'])
    
    return LUE_smooth_interpolation

# Water table ----------------------------------------
def process_WT():
    "Function to process Water Table related data"

    def WT_function(WT_cm):
        "WaterTable function calculation"
        value = 0.00033 * WT_cm^2 - 0.0014 * WT_cm + 0.75 # TODO -> Check if this works
        value[value[WT_cm] > 0] = value[value["WT_cm"] <= 0].max()
        value[value > 1] = 1 # TODO -> Check if this works
        return value
    
    WaterTable = pd.read_excel("data/OL1/OL1Data.xlsx", sheet_name="WaterTable")
    WaterTable["DateTime"] = WaterTable["DateTime"].apply(lambda x: pd.to_datetime(x, utc=True))
    WaterTable["DOY"] = WaterTable["DateTime"].dt.dayofyear
    WaterTable["WT_cm"] = WaterTable["WaterStand"] * 100
    WaterTable["f_WT"] = WaterTable["WT_cm"].apply(WT_function)

    # Linear interpolation of WT data 
    lowess = sm.nonparametric.lowess(WaterTable["f_WT"], WaterTable["DOY"], frac=0.5)  # frac is the smoothing parameter
    WT_smooth_interpolation = pd.DataFrame(lowess, columns=['DOY', 'WT_smoothed'])
    
    return WT_smooth_interpolation

def kalman_smoothing(data):
    model = UnobservedComponents(data, level='local level')
    result = model.fit()
    # Get smoothed (filled) values
    filled_data = result.smoothed_state[0]
    return pd.Series(filled_data, index=data.index)

def f_Tk_calculation(R, Ha, Hd, T_air):
    "Calculates f_Tk value"
    f_Tk = (Hd * math.exp(Ha * (T_air - T_opt) / (T_air * R/1e3 * T_opt))) / (Hd - Ha * (1 - math.exp(Hd * (T_air - T_opt) / (T_air * R/1e3 * T_opt))))
    return f_Tk

def GPP_calculation(DOY, temp, PAR, params):
    "Calculate GPP value"
    #Constants
    T_air = temp + 273          #Kelvin
    R = 8.134                   #Gas constant
    #Parameters
    Ha = params["Ha"]
    Hd = params["Hd"]
    k = params["k"]

    # Include variables that change seasonally
    LAI = LAI_smooth_func(DOY=DOY)
    LUE = LUE_smooth_func(DOY=DOY)
    
    # GPP Calculations
    f_Tk = f_Tk_calculation(R=R, Ha=Ha, Hd=Hd, T_air=T_air)
    f_par = 0.95*(1 - math.exp(-k * LAI))
    APAR = PAR * f_par

    #Return GPP value
    return ((LUE * APAR * f_Tk)/12)

def Reco_calculation(DOY, GPP, temp, params):
    "Calculate Reco value"
    #Constants
    T_air = temp + 273          #Kelvin
    R = 8.134                   #Gas constant
    #Parameters
    alpha_soc = params["alpha_soc"]
    ea_soc = params["EA_soc"]
    km_soc = params["kM_soc_avail"]
    alpha-lab = params["alpha_lab"]
    ea_lab = params["EA_lab"]
    km_lab = params["kM_lab"]
    C1_init = soc_total # μmolC/m3 Total Soil Organic Carbon 
    C2_init = 0         # C_laible, rapidly descomposed carbon
    # Convert temperature 
    RT = R * T_air 

    # Vmax calculations
    Vmax1 = alpha_soc * math.exp(-ea_soc / RT)
    Vmax2 = alpha_lab * math.exp(-ea_lab / RT)
    # C allocation based on GPP
    C2_in = (-1 * GPP * 60 * 30) * 0.5 # Allocate 50% of GPP to labile C pool

    # Initialize storage for results
    S1 =- len(DOY)
    S2 = len(DOY)
    Reco = len(DOY)
    f_WT = 1

    # Loop through time steps
    for t in range(1, len(DOY) +1):
        
        # C allocation
        if (t == 1):
            S1[t] = C1_init
            S2[t] = C2_init+C2_in[t]
        else:
            S1[t] = S1[t - 1]
            S2[t] = S2[t - 1]
        
        # Empirical factor for increased availability of SOC during early years
        S1[t] = S1[t] * k_soc 
        # Reaction velocities
        R1 = Vmax1[t] * S1[t] / (km_soc + S1[t])
        R2 = Vmax2[t] * S2[t] / (km_lab + S2[t])
        # Update SOC and labile C pools
        S1[t] = S1[t] - (R1)
        S2[t] = S2[t] - (R2) 
        # Reco calculations
        Reco[t] = (R1 + R2)*f_WT

def ghg_model(DOY, temp, PAR, params):
    "Function to define the ghg model"
    GPP = GPP_calculation(DOY, temp, PAR, params)
    Reco = Reco_calculation()
    NEE = Reco - GPP
    return pd.DataFrame({"DOY": DOY,
                             "Temp": temp,
                             "PAR": PAR,
                             "LAI": LAI,
                             "LUE": LUE,
                             "f_WT": f_WT,
                             "GPP": GPP,
                             "Reco": Reco, 
                             "NEE": NEE,})

if __name__ == "__main__":
    params1 = {"alpha_soc": alpha_soc*1e-1,
               "alpha_lab": alpha_lab*1e-1,
               "EA_soc": EA_soc,
               "EA_lab": EA_lab,
               "kM_soc_avail": kM_soc_avail, 
               "kM_lab": kM_lab, 
               "Ha": Ha*9e-1, 
               "Hd": Hd*9e-1, 
               "k": k, 
               "k_soc": k_soc}
    #TODO -> JOIN BOTH DFs IN ONE DF WITH THE NECESSARY DATA
    OL1_data_EC = pd.read_excel("data/OL1/OL1Data.xlsx", sheet="EC")
    NEE_fill = kalman_smoothing(OL1_data_EC["NEE_CO2"])
    Temp_fill = kalman_smoothing(OL1_data_EC["Temp"])
    OL1_data_PAR = pd.read_excel("data/OL1/OL1Data.xlsx", sheet="PAR")
    PAR_fill = kalman_smoothing(OL1_data_PAR["PAR"])
    OL1_data_EC = OL1_data_EC[OL1_data_EC["DOY"].notna() &
                               OL1_data_EC["PAR"].notna()]
    OL1_data_PAR = OL1_data_PAR[OL1_data_PAR["PAR"].notna()]

    ghg_df = ghg_model(DOY=OL1_data_EC["DOY"],
                       temp=OL1_data_EC["Temp"], 
                       PAR=OL1_data_PAR["PAR"], 
                       params=params1)
    
    # params1<-c(Ha=50e3,Hd=200e3, alpha_soc=0.5,alpha_lab=5,EA_soc=60e3,EA_lab=50e3,
    #        kM_soc_avail =kM_soc_avail ,kM_lab =kM_lab,
    #        k=k,k_soc)
    fit_pars = {"Ha":50e3,"Hd":200e3, 
             "alpha_soc ": 0.5,"EA_soc":60e3,
             "alpha_lab":5,"EA_lab":50e3,
             "kM_soc_avail":kM_soc_avail,"kM_lab":kM_lab}
    
    cost = cost_calculation(fit_pars=fit_pars,
                            data=OL1_data_EC,
                            response="NEE_fill")

    sens = sensFun(cost, fit_pars, OL1_data_EC, "NEE_fill")
    scatter_matrix(sens, alpha=0.8, figsize=(8, 8), diagonal='hist')
    # plt.show()
    ident = collin(sens)
    ident = ident[ident["collinearity"] > 20]

    fit2 = modFit(f = cost,
                  p = params1[["Hd","alpha_lab"]],
                  data = OL1_data_EC,
                  response = "NEE_CO2")
    
    # params1[names(coef(fit2))]<-coef(fit2)
    # model.output_fit = ghg_model(DOY=OL1_data_EC["DOY"],
    #                              PAR=OL1_data_EC["PAR"],
    #                              Temp=OL1_data_EC["Temp"],
    #                              params = params1)

    fit3 = modFit(f = cost,
                  p = params1[["Ha","EA_lab"]],
                  data = OL1_data_EC,
                  response = "NEE_fill")
    # params1[names(coef(Fit3))] = coef(Fit3)
    # model.output_fit = model(DOY=OL1Data["DOY"],
    #                          PAR=OL1Data["PAR"],
    #                          Temp=OL1Data["Temp"],
    #                          params = params1)