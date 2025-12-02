CLIMATE = {
    # General climate change terms
    'General climate change': """(
           climat*
        OR "global warming"
        OR "greenhouse effect*"
    )""",
    # Greenhouse gasses, including short-lived greenhouse gasses, when linked to emission or mitigation. Some astronomy results are filtered out.
    # Including direct and indirect climate forcers.Note: the combination with W/2 (emit* OR emission OR releas* OR mitigat*) is too restrictive.
    'Greenhouse gasses': """(
        (
              "carbon dioxide"
            OR co2
            OR methane
            OR ch4
            OR "nitrous oxide"
            OR n2o
            OR NO2
            OR "nitric oxide"
            OR "nitrogen dioxide"
            OR nox
            OR *chlorofluorocarbon*
            OR *cfc*
            OR refrigerant
            OR hydrofluorocarbon*
            OR hfc*
            OR *chlorocarbon*
            OR "carbon tetrachloride"
            OR ccl4
            OR halogen*
            OR ozone
            OR o3
            OR ammonia
            OR nh3
            OR "carbon monoxide"
            OR co
            OR "volatile organic compounds"
            OR nmvoc
            OR "hydroxyl radical"
            OR oh
            OR aerosol
            OR "black carbon"
            OR "organic carbon"
            OR "sulphur dioxide"
            OR "oxidized sulphur"
            OR so2
            OR sox
            OR "sulphuric acid"
            OR so4*
            OR sulfate*
            OR "black carbon"
            OR "fluorinated gas*"
            OR "particulate matter"
            OR pm10
            OR "pm 10"
          #  OR "pm2.5"  <-- one of these produces errors (?)
          #  OR "pm 2*”
          #  OR pm2*  <-- redundant
            OR "carbon emissions"
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
        OR "heavy rain*"
        OR "heat ind*"
        OR "extreme heat"
        OR "heat wave*"
        OR heatwave*
        OR "heat related"
        OR "extreme cold*"
        OR "cold ind*"
        OR "cold wave*"
        OR "cold spell*"
        OR humidity
        OR drought*
        OR hydroclim*
        OR monsoon
        OR "el ni$o"
        OR enso
        OR SOI
        OR "sea surface temperature*"
        OR sst
        OR "meteorological condition*"
        OR "meteorological data"
        OR "meteorological factor*"
        OR "meteorological indicator*"
        OR "meteorological variable*"
        OR "weather condition*"
        OR "weather factor*"
        OR "weather indicator*"
        OR "weather variable*"
        OR "weather related"
    )""",
    # Complex climate indices, including extreme weather events, floods, wildfire, and coastal changes. Some paleo-climatic events are excluded.
    'Complex climate indices': """(
           snowmelt*
        OR flood*
        OR storm*
        OR cyclone*
        OR hurricane*
        OR typhoon*
        OR "sea level"
        OR wildfire*
        OR "wild fire*"
        OR "forest fire*"
        OR river*
        OR "coast* erosion"
        OR "coastal change*"
        OR (
            (
                    extreme
                W/1 event*
            )
            AND NOT (
                paleo*
            )
        )
        OR (
            disaster*
            W/1 (
                   risk
                OR manag*
                OR natural
                OR prepare*
                OR prevent*
            )
        )
    )""",
    # Fossil fuels
    'Fossil fuels': """(
           "fossil fuel*"
        OR coal
        OR oil
        OR petroleum
        OR "natural gas"
        OR LNG
    )""",
    # Activities that produce climate forcers
    'Climate drivers': """(
           "energy production"
        OR "energy use"
        OR "energy consumption"
        OR "electricity production"
        OR "electricity generation"
        OR "steel production"
        OR heating
        OR cooking
        OR transport*
        OR gasoline
        OR diesel
        OR "jet fuel*"
        OR "shipping fuel*"
        OR industr*
        OR agricultur*
        OR waste
        OR building*
        OR fertilizer*
        OR "meat consum*"
        OR "consume meat*"
    )""",
    # Activities that reduce climate forcers
    'Mitigation': """(
           "energy transition"
        OR "renewable*"
        OR "clean energy"
        OR "emission control*"
        OR "particle filter*"
        OR reforestation
        OR "livestock management"
        OR "reduction of meat"
        OR "plant-based"
        OR "planetary health diet"
        OR "energy efficien*"
        OR recycl*
        OR "improved wastewater treatment"
        OR insulation
        OR "heat pump*"
        OR "solar panel*"
        OR "solar energy"
        OR "alternative fuel*"
        OR "carbon pricing"
        OR “carbon capture”
        OR "carbon storage"
        OR CCS
        OR "carbon dioxide removal"
        OR CDR
        OR "grid modernization"
        OR "active travel"
        OR "active mobility"
        OR cycling
        OR bike
        OR bikes
        OR bicycle*
        OR "electric vehicle*"
        OR "electric car*"
        OR "mobility shift*"
        OR "carbon neutral*"
        OR "low carbon"
        OR decarb*
        OR "green infrastructure*"
        OR "green urban"
        OR "green space"
        OR greenspace
        OR "nature-based"
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
        OR ill
        OR illness
        OR disease*
        OR syndrome*
        OR infect*
        OR medical*
        OR medicine
    )""",
    # General health outcomesNote: Unclear what “{a&e}” stands for; also added “intensive care”, ICU” fatalities
    'General health outcomes': """(
           mortality
        OR daly
        OR dalys
        OR morbidity
        OR injur*
        OR death*
        OR acciden*
        OR epidemic*
        OR pandemic*
        OR wound*
        OR hospital*
        OR medical cent*
        OR clinic*
        OR fatalities
        OR emergency
        OR emergencies
        OR ICU
        OR "intensive care"
        OR "stroke unit"
        OR doctor
        OR doctors
        OR clinician*
        OR surgeon*
        OR surger*
        OR "general practi*"
        OR nurs*
    )""",
    # Nutrition/food quality and quantity, including obesity and undernutrition Note: this can be both outcome and exposure
    'Nutrition/food quality': """(
           obes*
        OR overweight "overweight"
        OR underweight
        OR "underweight"
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
    )""",
    # Cardio-vascular terms. Some studies on Chemical Vapour Deposition (CVD) are excluded.
    'Cardio-vascular': """(
           hypertens*
        OR "blood pressure"
        OR stroke
        OR *vascular
        OR "heart disease"
        OR isch?emic
        OR cardiovascular
        OR "cardio vascular"
        OR "heart attack*"
        OR coronary
        OR chd
        OR infarct*
        OR (
            cvd
            AND NOT (
                   vapour
                OR vapor
            )
        )
    )""",
    # Renal health terms and cancer  Note: added “calculi/calculus”
    'Renal health': """(
           ckd
        OR renal
        OR cancer
        OR neoplasm*
        OR tumor*
        OR tumour*
        OR kidney
        OR calculus
        OR calculi
        OR lithogenes*
    )""",
    # Effects of temperature extremesNote: added “thermal stability” (for heat effects on drugs)
    'Effects of temperature extremes': """(
        (
            heat 
            W/2 (
                   stress
                OR fatigue
                OR burn*
                OR stroke
                OR exhaustion
                OR cramp*
            )
        )
        OR skin
        OR fever*
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
        OR stillbirth
        OR birth?weight
        OR maternal
        OR pregnan*
        OR gestation*
        OR *eclampsia
        OR sepsis
        OR oligohydramnios
        OR placenta*
        OR haemorrhag*
        OR hemorrhag*
    )""",
    # Vector-borne diseasesNote: added "rift valley" OR "ross river" (both diseases)
    'Vector-borne diseases': """(
           malaria
        OR dengue*
        OR mosquito*
        OR chikungunya
        OR leishmaniasis
        OR encephalit*
        OR "vector borne"
        OR vectorborne
        OR pathogen*
        OR zoonos*
        OR zika
        OR "west nile"
        OR onchocerciasis
        OR filiariasis
        OR lyme
        OR "tick borne"
        OR tickborne
        OR "rift valley"
        OR "ross river"
    )""",
    # Bacterial, parasitic and viral infections, including waterborne and foodborne diseasesNote: also added cryptosporidiosis, leptospirosis, typhoid, melioidosis, "hepatitis E", dysentery
    'Bacterial, parasitic and viral infections': """(
           waterborne
        OR "water borne"
        OR diarrhoea*
        OR diarrh*
        OR gastro*
        OR enteric
        OR *bacteria*
        OR viral
        OR *virus*
        OR parasit*
        OR vibrio*
        OR cholera*
        OR "e.coli"
        OR "Escherichia coli"
        OR protozoa*
        OR salmonel*
        OR giardia
        OR shigella
        OR campylobacter
        OR "food borne"
        OR foodborne
        OR aflatoxin
        OR poison*
        OR ciguatera
        OR "algal bloom*"
        OR cryptosporidiosis
        OR leptospirosis
        OR typhoid
        OR melioidosis
        OR "hepatitis E"
        OR dysentery 
        OR (
            (
                  snake*
               OR adder*
            ) 
            W/2 bite*
        )
    )""",
    # Air quality and allergens
    'Air quality and allergens': """(
           "air pollution"
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
    # Respiratory outcomesNote: also added bacterial meningitis
    'Respiratory outcomes': """(
           respiratory
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
    # Mental health outcomesNote: also added psychiat*
    'Mental health outcomes': """(
           mental
        OR depress*
        OR *stress*
        OR anxi*
        OR ptsd
        OR psycho*
        OR *trauma*
        OR suicid*
        OR solastalgi*
        OR psychiatr*
    )""",
    # Water quality and quantity
    'Water quality and quantity': """(
           "water security"
        OR "water insecurity"
        OR "water quantity"
        OR "water scarcity"
        OR "water quality"
        OR "contaminated water"
        OR "water contamination*"
    )""",
    # Social factors and vulnerability
    'Social factors and vulnerability': """(
           migrat*
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
    )""",
    # Health systems
    # 'Health systems': '''[no additional terms needed]''',
}

MERGED = {
    'CLIMATE AND HEALTH': '(' + (' OR '.join(CLIMATE.values())) + ') AND (' + (' OR '.join(CLIMATE.values())) + ')',
    'CLIMATE': ' OR '.join(CLIMATE.values()),
    'HEALTH': ' OR '.join(HEALTH.values()),
}
