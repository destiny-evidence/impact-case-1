# This contains an attempt to translate the solr query to a superset query that runs
# on the OpenAlex API. Mainly this expands as many wildcards as possible and replaces
# NEAR/ with AND.


# "heat wave" "heat-wave" "heat waves" seem to be the same to openalex but not heatwave
# british and english spelling not normalised, needs both
# openalex uses stemming, so declinations do not need to be "wildcarded"

# climat* OR "global warming" OR "greenhouse effect" OR "greenhouse effects" OR "greenhouse gas" OR
# "greenhouse gases" OR "greenhouse gas emissions" OR "greenhouse emissions" OR "GHG emissions" OR
# "GHGE" OR temperature* OR precipitat* OR rainfall OR "heat index" OR "heat indices" OR
# "extreme heat event" OR "extreme heat events" OR "heat-wave" OR heatwave OR "extreme-cold*" OR
# "cold index" OR "cold indices" OR humidity OR drought* OR hydroclim* OR monsoon OR "el nino" OR
# ENSO OR "sea surface temperature" OR "sea surface temperatures" OR SST OR snowmelt* OR flood* OR
# storm* OR cyclone* OR hurricane* OR typhoon* OR "sea-level" OR "sea level" OR wildfire* OR
# "wild-fire" OR "forest-fire" OR "forest fire" OR "forest fires"
# TODO: extend climat*
# TODO: verify precipitat*
# TODO: verify drought*
# TODO: verify flood*
# TODO: verify storm*
q1 = """
climate OR "global warming" OR "greenhouse effect" OR "greenhouse gas" OR "greenhouse emission" OR
GHG OR temperature OR precipitation OR rainfall OR "heat index" OR "heat indices" OR
"extreme heat event" OR "heat wave" OR heatwave OR "extreme cold" OR "cold index" OR "cold indices" OR
humidity OR drought OR hydroclimate OR hydroclimatic OR hydroclimatology OR hydroclimatological OR
monsoon OR "el nino" OR ENSO OR "sea surface temperature" OR SST OR snowmelt OR flood OR storm OR cyclone OR
hurricane OR typhoon OR "sea level" OR wildfire OR "wild fire" OR "forest fire"
"""

# health OR wellbeing OR "well-being" OR ill OR illness OR disease* OR syndrome* OR infect* OR
# medical* OR mortality OR DALY OR morbidity OR injur* OR death* OR hospital* OR acciden* OR
# emergency OR emergent OR doctor OR GP OR obes* OR overweight OR "over-weight" OR underweight
# OR "under-weight" OR hunger OR stunting OR wasting OR undernourish* OR undernutrition OR
# anthropometr* OR malnutrition OR malnour* OR anemia OR anaemia OR "micro-nutrient*" OR
# hypertension OR "blood pressure" OR stroke OR renovascular OR cardiovascular OR cerebrovascular OR
# "heart disease" OR Isch*emic OR cardio*vascular OR "heart attack" OR
# "heart attacks" OR coronary OR CHD OR diabet* OR CKD OR renal OR cancer OR kidney OR lithogenes* OR
# skin OR fever* OR renal* OR rash* OR eczema* OR "thermal stress" OR hypertherm* OR hypotherm* OR
# pre*term OR stillbirth OR birth*weight OR LBW OR maternal OR pregnan* OR gestation* OR "pre-eclampsia" OR
# "preeclampsia" OR sepsis OR oligohydramnios OR placenta* OR haemorrhage OR hemorrhage OR malaria OR dengue* OR
# mosquito* OR chikungunya OR leishmaniasis OR encephalit* OR vector-borne OR pathogen OR zoonos* OR zika* OR
# "west nile" OR onchocerciasis OR filiariasis OR waterborne OR diarrhoeal OR diarrheal OR gastro*
# OR "vibrio bacteria" OR cyanobacteria OR parasit* OR borrelia OR paraly* OR neurotoxi* OR viral OR
# rotavirus OR noravirus OR hantavirus OR cholera OR protozoa* OR lyme OR tick*borne OR salmonella OR
# giardia OR shigella OR campylobacter OR food*borne OR aflatoxin OR poison* OR ciguatera OR respiratory
# OR allerg* OR lung* OR asthma* OR bronchi* OR pulmonary* OR COPD OR rhinitis OR wheez* OR mental OR
# depress* OR anxiety OR PTSD OR psycho* OR suicide* OR
#  "pre-trauma" OR "pre trauma" OR pretrauma "post-trauma" OR "post trauma" or posttrauma
# OR (CVD NOT (vapor OR vapour))
# TODO: check infect*
# TODO: check hospital*
# TODO: check parasit*
# TODO: check bronchi*
# TODO: check psycho*
q2 = """
health OR wellbeing OR "well being" OR ill OR illness OR disease OR syndrome OR infect OR infection OR infectious OR
medical OR mortality OR DALY OR morbidity OR injury OR death OR hospital OR hospitalization OR hospitalisation OR
accident OR accidental OR emergency OR emergent OR doctor OR GP OR obesity OR obese OR overweight OR "over weight" OR
underweight OR "under weight" OR hunger OR stunting OR wasting OR undernourishment OR undernourish OR
undernutrition OR anthropometric OR anthropometry OR malnutrition OR malnourishment OR malnourish OR
anemia OR anaemia OR "micro nutrient" OR hypertension OR "blood pressure" OR stroke OR renovascular OR
cerebrovascular OR "heart disease" OR cardiovascular OR "cardio vascular" OR "heart attack" OR
ischemic OR ischaemic OR coronary OR CHD OR diabetes OR diabetic OR CKD OR renal OR cancer OR
kidney OR lithogenesis OR lithogenes OR skin OR fever OR feverish OR renal OR rash OR eczema OR
eczematous OR "thermal stress" OR hyperthermia OR hyperthermic OR hypothermia OR hypothermic OR
preterm OR stillbirth OR "birth weight" OR LBW OR maternal OR pregnant OR pregnancy OR gestation OR "pre eclampsia" OR
preeclampsia OR sepsis OR oligohydramnios OR encephalitis OR encephalitic OR
placenta OR haemorrhage OR hemorrhage OR malaria OR dengue OR mosquito OR chikungunya OR leishmaniasis OR
"vector borne" OR pathogen OR zoonosis OR zoonose OR zika OR "west nile" OR onchocerciasis OR filiariasis OR
waterborne OR diarrhoeal OR diarrheal OR gastrointestinal OR gastroenterology OR gastroesophageal OR
"vibrio bacteria" OR cyanobacteria OR parasitic OR borrelia OR paralysis OR paralyzed OR paralytic OR
paralysed OR neurotoxic OR neurotoxicity OR neurotoxin OR viral OR rotavirus OR noravirus OR hantavirus OR cholera OR
protozoa OR protozoan OR protozoal OR lyme OR "tick borne" OR salmonella OR giardia OR shigella OR campylobacter OR
"food borne" OR aflatoxin OR poison OR poisonous OR ciguatera OR respiratory OR allergic OR allergen OR allergy OR
lung OR asthma OR asthmatic OR bronchial OR bronchitis OR pulmonary OR COPD OR rhinitis OR wheezing OR wheeze OR
mental OR depression OR depressive OR depressed OR anxiety OR PTSD OR psychological OR psychosocial OR psychometric OR
psychotic OR suicide OR "pre trauma" OR pretrauma OR "post trauma" OR posttrauma OR (CVD NOT (vapor OR vapour))
"""

query = f"""
(
    {q1}
    OR
    (disaster AND (risk OR management OR manage OR managing OR natural))
    OR
    ((extreme AND event) NOT paleo)
    OR
    (
        (
            hydrochloroflourocarbon OR pm2.5 OR ammonia OR nox OR
            HFC OR SO4 OR carbon OR n20 OR halogen OR chlorocarbon OR nh3 OR SOX OR O3 OR ccl4 OR
            NMVOC OR SO2 OR HFC OR CO OR nitrous OR methane OR ch4 OR co2 OR sulphur OR VOC OR ozone OR chlorocarbons
        )
        AND
        (emit OR mitigate OR emission OR mitigation)
    )
)
AND
(
    {q2}
    OR (
        enteric
        NOT (fermentation OR "enteric CH4" OR "enteric methane")
    )
    OR (
        heat
        AND
        (stress OR fatigue OR burn OR stroke OR exhaustion OR cramp)
        NOT cattle
    )
)
"""
