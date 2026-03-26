CLIMATE = {
    # General climate change terms
    'General climate change': """(
           climate
	OR climatic
	OR climatically
	OR climatology  
	OR climatogenic
        OR "global warming"
        OR "greenhouse effect" 
	OR "greenhouse effects"      
    )""",
    # Greenhouse gasses, including short-lived greenhouse gasses, when linked to emission or mitigation. Some astronomy results are filtered out.
    # Including direct and indirect climate forcers.
    'Greenhouse gasses': """(
        (
                "greenhouse gas"
	     OR "greenhouse gases"			
             OR "carbon dioxide"
             OR co2
             OR methane
             OR ch4
             OR "nitrous oxide"
             OR n2o 
	     OR no2
             OR "nitric oxide"
             OR "nitrogen dioxide"
             OR nox
             OR chlorofluorocarbon
	     OR chlorofluorocarbons	
             OR cfc
             OR refrigerant
             OR hydrofluorocarbon
	     OR hydrofluorocarbons
             OR hfc
             OR chlorocarbon
       	     OR chlorocarbons
             OR "carbon tetrachloride"
             OR ccl4
             OR halogen
	     OR halogens
             OR ozone
             OR o3
             OR ammonia
             OR nh3
             OR "carbon monoxide"
             OR "volatile organic compounds"
             OR VOCs
             OR nmvoc
             OR "hydroxyl radical"
             OR "oh"
             OR aerosol
	     OR aerosols
             OR "black carbon"
             OR soot 
	     OR "organic carbon"
             OR "sulfur dioxide"
             OR "sulphur dioxide"
             OR "oxidized sulfur"
             OR "oxidized sulphur"
             OR so2
             OR sox
             OR "sulphuric acid"
             OR "sulfuric acid"
             OR so4
	     OR sulfate
	     OR sulfates
             OR "fluorinated gas"
	     OR "fluorinated gases"    
             OR "particulate matter"
             OR pm10
             OR "pm 10"
	     OR pm2	
	     OR "pm 2"	
             OR "pm2.5"
             OR "pm 2.5"
  	     OR "pm25"
             OR "pm 25"
             OR "carbon emissions"
             OR "ghg emissions"
             OR "climate forcer"
	     OR "climate forcers"     
             OR slcf
             OR slcfs
        )
        AND NOT (
             star
             OR "solar system"
        )
    )""",
    # Climate variability indicators/climate indices
    'Climate variability': """(
           temperature*
        OR precipitat*
        OR rainfall*
        OR "heavy rain*"      # alternatives ("heavy rain", "heavy rains", "heavy rainfall", "heavy rainfalls", "heavy raining",""heavy rainings","heavy rainstorm", "heavy rainstorms")
        OR "extreme rain*"    # alternatives ("extreme rain", "extreme rains", "extreme rainfall", "extreme rainfalls", "extreme raining",""extreme rainings","extreme rainstorm", "extreme rainstorms")
        OR "heat ind*"        # alternatives ("heat index","heat indexes", "heat indexs")
        OR "extreme heat"
        OR "heat wave*"       # alternatives ("heat wave","heat waves")
        OR heatwave*          # alternatives ("heatwave","heatwaves")
        OR "heat related"
        OR "urban heat"
        OR "extreme cold*" # alternatives ("extreme cold","extreme colds","extreme coldness")
        OR "cold index*" # alternatives ("cold index","cold indexes", "cold indexs")
        OR "cold indic*" # alternatives("cold indicator","cold indicators","cold indices","cold indicative")
        OR "cold induced"
        OR "cold wave*" # alternatives("cold wave", "cold waves")
        OR "cold spell*" # alternatives ("cold spell", "cold spells")
        OR humidity
        OR drought*
        OR hydroclim*
        OR monsoon
        OR "el nino"
        OR "el niño"
        OR enso
        OR "southern oscillation index"
        OR "la nina"
        OR "la niña"
        OR "sea surface temperature*" # alternatives ("sea surface temperature","sea surface temperatures")
        OR "meteorological condition*"  # alternatives("meteorological condition","meteorological conditions")
        OR "meteorological data"
        OR "meteorological factor*"  # alternatives("meteorological factor","meteorological factors")
        OR "meteorological indicator*"  # alternatives("meteorological indicator","meteorological indicators")
        OR "meteorological variable*"  # alternatives("meteorological variable","meteorological variables")
        OR "weather condition*" # alternatives ("weather condition","meteorological conditions")
        OR "weather factor*" #  alternatives ("weather factor", "weather factors")
        OR "weather indicator*" # alternatives ("weather indicator", "wweather indicators")
        OR "weather variable*" # alternatives ("weather variable", "weather variables")
        OR "weather related"
        OR "weather induced"
        OR "climatic extreme*"   # alternatives ("climatic extreme", "climatic extremes")
    )""",
    # Complex climate indices, including extreme weather events, floods, wildfire, and coastal changes. Some paleo-climatic events are excluded.
    #
    'Complex climate indices': """(
           snowmelt*
        OR landslide*
        OR mudslide*
        OR flood*
        OR storm*
        OR cyclone*
        OR hurricane*
        OR typhoon*
        OR "sea level*"# alternatives ("sea level", "sea levels")
        OR sealevel*
        OR wildfire*
        OR "wild fire*" # alternatives ("wild fire", "wild fires")
        OR bushfire*
        OR "bush fire*"
        OR "forest fire*" # alternatives ("forest fire", "forest fires")
        OR river*
        OR (
            ("extreme event*") # alternatives ("extreme event", "extreme events")
            AND NOT paleo*
        )
        OR "extreme weather event*" # alternatives ("extreme weather event", "extreme weather events")
        OR "coast erosion"
        OR "coastal erosion"
        OR "coastal change*" # alternatives ("coastal change", "coastal changes")
        OR "coastal inundation*" # alternatives ("coastal inundation", "coastal inundations")
        OR "saltwater intrusion"
        OR "natural disaster*" # alternatives ("natural disaster", "natural disasters")
        OR "climate disaster*" # alternatives ("climate disaster", "climate disasters")
        OR "climate related disaster*" # alternatives ("climate related disaster", "climate relatd disasters")
        OR "climate hazard*"  # alternatives ("climate hazard", "climate hazards")
        OR "climate related hazard*" # alternatives ("climate related hazard", "climate related hazards")
        OR "weather related disaster*" # alternatives ("weather related disaster", "weather related disasters")
        OR "climate related disaster*" # alternatives ("climate related disaster", "climate related disasters")
        OR "weather driven disaster*"  # alternatives ("weather driven disaster", "weather driven disasters")
        OR "climate driven disaster*" # alternatives ("climate driven disaster", "climate driven disasters")
    )""",
    # Fossil fuels
    'Fossil fuels': """(
          "fossil fuel*" # alternatives ("fossil fuel", "fossil fuels","fossil fueling")
        OR coal
        OR oil
        OR petrol*
        OR "natural gas"
        OR LNG
    )""",
    # Activities that produce climate forcers
    'Climate forcers production': """(
          "energy production"
       OR "energy use"
       OR "energy consumption"
       OR "electricity production"
       OR "electricity generation"
       OR "power generation"
       OR "steel production"
       OR "concrete production"
       OR heating
       OR cooling
       OR "air condition*"
       OR refrigerat*
       OR cooking
       OR transport*
       OR gasoline
       OR diesel
       OR "jet fuel*" # alternatives ("jet fuel", "jet fuels","jet fueling")
       OR "shipping fuel*" # alternatives  ("shipping fuel", "shippingfuels","shipping fueling")
       OR industr*
       OR agricultur*
       OR waste
       OR building*
       OR fertilizer*
       OR "meat consum*" # alternatives  ("meat consumption", "meat consumer","meat consumers", "meat consumptions", "meat consumed")
       OR "consume meat"
       OR beef
       OR "red meat"
       OR "agricultural livestock production"
       OR "livestock diet modification*" # alternatives  ("livestock diet modification", "livestock diet modifications")
       OR "methane leak detection"
       OR "biomass burning"
       OR deforest*
       OR aviation
       OR "marine shipping"
       OR "open burning"
       OR "gas flaring"
       OR "road construction"
       OR "asphalt production"
       OR "construction material*" # alternatives ( "construction material", "construction materials")
       OR cement
    )""",
    # Activities that reduce climate forcers
    'Mitigation': """(
           "energy transition"
        OR renewable*
        OR "clean energy"
        OR "emission control*"
        OR "particle filter*"
        OR reforestation
        OR afforestation
        OR "livestock management"
        OR "livestock manure management"
        OR "reduction of meat"
        OR "plant based"
        OR "sustainable diet"
        OR "plant rich diet*"
        OR "planetary health diet"
        OR "food waste"
        OR "circular economy"
        OR "energy efficien*"
        OR recycl*
        OR reuse
        OR reusing
        OR "improved wastewater treatment"
        OR "improved agricultural practic*"
        OR "precision farming"
        OR insulation
        OR "heat pump*"
        OR "solar panel*"
        OR "solar energy"
        OR "solar system*"
        OR "alternative fuel*"
        OR "carbon pricing"
        OR "carbon capture"
        OR "carbon storage"
        OR "carbon dioxide removal"
        OR "grid moderni*"
        OR "active travel*"
        OR "active mobility"
        OR cycling
        OR bike
        OR bikes
        OR bicycle*
        OR "electric vehicle*"
        OR "electric car*"
        OR "electric mobility"
        OR "mobility shift*"
        OR "carbon neutral*"
        OR "low carbon"
        OR decarb*
        OR "green infrastructure*"
        OR "green urban"
        OR "urban green*"
        OR "green space"
        OR greenspace
        OR "nature based"
        OR sustainab*
        OR "emission reduc*"
        OR "reduce emission*"
        OR "reduction of emission*"
        OR "carbon reduc*"
        OR "reduce carbon"
        OR "reduction of carbon"
        OR "carbon footprint"
    )""",
}
HEALTH = {
    # General health terms
    'General health terms': """(
           health*
        OR "well being"
        OR wellbeing
        OR medical*
        OR medicine
        OR clinic*
        OR hospital*
        OR fatalities
        OR emergenc*
        OR ICU
        OR "intensive care"
        OR "stroke unit*"
        OR doctor
        OR doctors
        OR clinician*
        OR surgeon*
        OR surger*
        OR "general practi*"
        OR nurs*
        OR "primary care*"
        OR "secondary care*"
        OR "tertiary care*"
    )""",
    # General health outcomes
    #
    # Note: Unclear what "{a&e}" stands for; also added "intensive care", ICU" fatalities
    'General health outcomes': """(
           mortality
        OR daly
        OR dalys
        OR disease*
        OR morbid*
        OR injur*
        OR death*
        OR acciden*
        OR epidemic*
        OR pandemic*
        OR wound*
        OR burn*
        OR "quality of life"
        OR qol
        OR hrqol
        OR ill
        OR illness
        OR syndrome*
        OR infect*
    )""",
    # Nutrition/food quality and quantity, including obesity and undernutrition
    # Note: this can be both outcome and exposure
    'Nutrition/food quality': """(
           obes*
        OR overweight
        OR "over weight"
        OR underweight
        OR "under weight"
        OR hunger
        OR stunt*
        OR wasting
        OR undernourish*
        OR undernutrition
        OR anthropometr*
        OR malnutrition
        OR malnour*
        OR anemia
        OR anaemia
        OR "micronutrient*"
        OR "micro nutrient*"
        OR diabet*
        OR T2D*
        OR T1D*
        OR "food insecurity"
        OR "food security"
        OR "food production"
        OR "food scarcity"
        OR "food supply"
        OR "food system*"
        OR "food quality"
        OR "food quantity"
        OR "food industry"
    )""",
    # Cardio-vascular terms. Some studies on Chemical Vapour Deposition (CVD) are excluded.
    'Cardio-vascular': """(
           hypertens*
        OR "blood pressure"
        OR stroke
        OR vascular
        OR macrovascular
        OR microvascular
        OR "heart disease"
        OR ischemic
        OR ischaemic
        OR cardiovascular
        OR "cardio vascular"
        OR "heart attack*"
        OR "coronary heart"
        OR infarct*
    )""",
    # Renal health terms and cancer  Note: added "calculi/calculus"
    'Renal health': """(
           renal
        OR kidney
        OR calculus
        OR calculi
        OR lithogenes*
        OR cancer
        OR neoplasm*
        OR tumor*
        OR tumour
    )""",
    # Effects of temperature extremes
    #
    # Note: added "thermal stability" (for heat effects on drugs)
    'Effects of temperature extremes': """(
           "heat stress"
        OR "heat illness*"
        OR "heat fatigue*"
        OR "heat burden*"
        OR "heat stroke*"
        OR "heat exhaustion"
        OR "heat cramp*"
        OR "heat syncope"
        OR skin
        OR dermal
        OR rash*
        OR eczema*
        OR "thermal stress*"
        OR hypertherm*
        OR hypotherm*
        OR "thermal stability"
    )""",
    # Maternal health outcomes
    'Maternal health outcomes': """(
           preterm
        OR "pre term"
        OR prematur*
        OR stillbirth*
        OR birthweight
        OR "birth weight"
        OR maternal
        OR pregnan*
        OR gestation*
        OR eclampsia
        OR preeclampsia
        OR sepsis
        OR oligohydramnios
        OR placenta*
        OR haemorrhag*
        OR hemorrhag*
    )""",
    # Vector-borne diseases
    # Note: added "rift valley" OR "ross river" (both diseases)
    'Vector-borne diseases': """(
           malaria
        OR dengue*
        OR mosquito*
        OR anopheles
        OR aedes
        OR culex
        OR chikungunya
        OR leishmaniasis
        OR encephalit*
        OR "vector borne"
        OR vectorborne
        OR pathogen*
        OR zoonos*
        OR zika
        OR "west nile"
        OR "crimean congo"
        OR onchocerciasis
        OR filiariasis
        OR "lyme disease"
        OR tick
        OR ticks
        OR tickborne
        OR "Ixodes ricinus"
        OR "Dermacentor reticulatus"
        OR "Dermacentor marginatus"
        OR "Hyalomma marginatum"
        OR "Haemaphysalis concinna"
        OR "rift valley"
        OR "ross river"
        OR "Usutu virus"
        OR tularemia
        OR "rabbit fever"
        OR "Q fever"
        OR "yellow fever"
    )""",
    # Bacterial, parasitic and viral infections, including waterborne and foodborne diseases
    # Note: also added cryptosporidiosis, leptospirosis, typhoid, melioidosis, "hepatitis E", dysentery
    'Bacterial, parasitic and viral infections': """(
           "water related"
        OR waterborne
        OR "water borne"
        OR diarrh*
        OR gastro*
        OR enteric
        OR bacteria*
        OR viral
        OR virus*
        OR arbovirus*
        OR norovirus
        OR rotavirus
        OR "barmah forest"
        OR lassa
        OR parasit*
        OR vibrio*
        OR cholera*
        OR "e.coli"
        OR "Escherichia coli"
        OR protozoa*
        OR salmonel*
        OR giardia*
        OR shigell*
        OR campylobacter*
        OR cryptosporid*
        OR legionell*
        OR "food related"
        OR "food borne"
        OR foodborne
        OR aflatoxin
        OR poison*
        OR ciguatera
        OR "algal bloom*"
        OR cryptosporidiosis
        OR leptospir*
        OR typhoid
        OR melioidosis
        OR "hepatitis A"
        OR "hepatitis E"
        OR dysentery
    )""",
    # Air quality and allergens
    'Air quality and allergens': """(
           air pollution"
        OR "air quality"
        OR smoke
        OR dust
        OR haze
        OR ambrosia
        OR ragweed
        OR mold*
        OR pollen
        OR spores
        OR pm10
        OR pm2*
    )""",
    # Respiratory outcomes
    'Respiratory outcomes': """(
           espiratory
        OR allerg*
        OR lung*
        OR asthma*
        OR bronchi*
        OR pulmonary
        OR copd
        OR rhinitis
        OR wheez*
        OR "bacterial meningitis"
    )""",
    # Mental health outcomes
    'Mental health outcomes': """(
           mental
        OR depress*
        OR stress*
        OR anxiet*
        OR ecoanxiet*
        OR ptsd
        OR psycho*
        OR trauma*
        OR suicid*
        OR solastalgi*
        OR psychiatr*
    )""",
    # Water quality and quantity
    'Water quality and quantity': """(
           "water security"
        OR "water suppl*"
        OR "water resources"
        OR "water insecurity"
        OR "water quantity"
        OR "water scarcity"
        OR "water quality"
        OR "contaminated water"
        OR "water contamination*"
    )""",
    # Social factors and vulnerability
    'Social factors and vulnerability': """(
           migra*
        OR displace*
        OR conflict*
        OR violen*
        OR homeless*
        OR poverty
        OR poor
        OR disadvantaged
        OR vulnerab*
        OR elder*
        OR frail*
        OR disab*
        OR "informal settlement*"
    )""",
    # Adaptation or mitigation
    'Adaptation or mitigation': """(
           adapt
        OR adapted
        OR adaptation
        OR mitigate
        OR mitigated
        OR mitigation
    )""",
}

# OR ( without combination with climate or health)
ADAPTATION = {
    # Adaptation
    'Adaptation': """(
           "climate resilien*"
        OR "health care resilience"
        OR "resilient health*"
        OR "healthcare resilience"
        OR "health system resilience"
        OR "health care system resilience"
        OR "healthcare system resilience"
        OR "eco friendly health*"
        OR "green health*"
        OR "circular health*"
        OR "low impact health*"
        OR "disaster risk reduction"
        OR "disaster manag*"
        OR "disaster prepare*"
        OR "disaster prevent*"
        OR "disaster risk prevent*"
        OR "heat prevent*"
        OR "heat protect*"
        OR "heat resilien*"
        OR "heat adapt*"
        OR "heat mitigat*"
        OR "heat stress prevent*"
        OR "heat action*"
        OR "heat stress management"
        OR "heat warning"
        OR "heat risk reduct*"
        OR "heat prepared*"
        OR "cooling area*"
        OR "cooling access*"
        OR "cooling shelter*"
        OR "cooling center*"
        OR "cooling centre*"
        OR "passive cooling"
        OR "urban cooling"
        OR "climate smart"
        OR "nature based solution*"
        OR "climate risk*"
        OR "wildfire risk reduction*"
        OR "bushfire risk reduction*"
    )""",
}

expansions = {
    # 'acciden': [],
}
MERGED = {
    'CLIMATE': ' OR '.join(CLIMATE.values()),
    'HEALTH': ' OR '.join(HEALTH.values()),
    'ADAPTATION': ' OR '.join(ADAPTATION.values()),
}
MERGED['(CLIMATE AND HEALTH) OR ADAPTATION'] = f'(({MERGED["CLIMATE"]}) AND ({MERGED["HEALTH"]})) OR {MERGED["ADAPTATION"]}'
