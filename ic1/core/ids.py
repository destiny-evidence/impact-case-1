"""Shared identifiers for the IC1 NACSOS project (single source of truth)."""

# The NACSOS project for impact case 1.
PROJECT_ID = 'db6ee519-afb5-4813-822b-bfbc7dfd2237'

# The IN/OUT (relevance screening) annotation scheme.
INOUT_SCHEME_ID = '0689d927-f78d-46aa-bbcf-190ce156f707'

# The DESTINY taxonomy annotation scheme (see ic1/annotation/scheme/import_taxonomy.py).
TAXONOMY_SCHEME_ID = '9d5d7fac-188c-4467-a1ad-021bebec69c1'

# SELECT ass.assignment_scope_id, ass.name, bam.bot_annotation_metadata_id, bam.name
# FROM assignment_scope ass
#      LEFT OUTER JOIN bot_annotation_metadata bam ON ass.assignment_scope_id = bam.assignment_scope_id
# WHERE ass.annotation_scheme_id = '0689d927-f78d-46aa-bbcf-190ce156f707';
INOUT_ANNOTATION_SCOPES: list[tuple[str, str]] = [
    (
        '4bb0f2b5-5552-44f4-8760-5eb9fd41c115',  # R 01 | G 01 | B 1/5 | EM,IK,NS | COMPLETE
        '24736ff3-d162-49f6-adf9-0bb0bcee1823',  # Resolved -> R 01 | G 01 | B 1/5 | EM,IK,NS (Andres)
    ),
    (
        '3690621f-f573-441d-b570-40ec9793873e',  # R 01 | G 01 | B 2/5 | EM,IK,NS | COMPLETE
        '4f39d574-3b2d-4c8f-960a-6d7ee7b3de52',  # Resolved -> R 01 | G 01 | B 2/5 | EM,IK,NS (Andres)
    ),
    (
        'a9e98a13-beed-4b84-9399-1454c1a82436',  # R 01 | G 01 | B 3/5 | EM,IK,NS | COMPLETE
        'aa24bf46-9a3f-4369-9d41-c0f575c59ce7',  # Resolved_R 01 | G 01 | B 3/5 | EM,IK,NS | (Andres)
    ),
    (
        '689ae557-2712-4ab7-b28b-9ebc7e872f69',  # R 01 | G 01 | B 4/5 | EM,IK+AR | COMPLETE
        '6e3cb281-234d-449d-8c3c-75241964d394',  # Resolved_R 01 | G 01 | B 4/5 | EM,IK+AR (Andres)
    ),
    (
        'b6e9a7de-65f8-4638-a03c-40b5769901b4',  # R 01 | G 01 | B 5/5 | EM,IK + AM | COMPLETE
        '8d96c0ba-70dc-4903-a7f8-469bfd3853b6',  # Resolved_R 01 | G 01 | B 5/5 | EM,IK + AM (Maria-Inti)
    ),
    (
        '2e4e2173-6a35-457f-aeda-013552b63fa6',  # R 01 | G 02 | B 1/5 | KH,PL+AM | COMPLETE
        '5b0e7ada-41f5-49ba-b5c4-7e261f20ace7',  # Resolved -> R 01 | G 02 | B 1/5 | KH,LM,PL+AM (Maria-Inti)
    ),
    (
        '3a3fda80-2a18-4558-9571-37e9a9a1c60b',  # R 01 | G 02 | B 2/5 | PL+AM,EM | COMPLETE
        'd8be345b-27bb-4320-8fc4-6091c73ada79',  # Resolved_R 01 | G 02 | B 2/5 | PL+AM,EM (Maria-Inti)
    ),
    (
        'bd7bca6d-126a-4c3c-baaf-df4abda99b77',  # R 01 | G 02 | B 3/5 | KH,PL,AM | COMPLETE
        '70c3d302-f9d5-4219-9501-f2b04945aab8',  # Resolved_R 01 | G 02 | B 3/5 | KH,PL,AM | (Lotte)
    ),
    (
        '580ea284-4941-45b5-a833-b8323c89a2c5',  # R 01 | G 02 | B 4/5 | KH,PL+AM,AR | COMPLETE
        'f4256501-9f36-4734-b092-735e01f67407',  # Resolved_R 01 | G 02 | B 4/5 | KH,PL+AM,AR | (Lotte)
    ),
    (
        'ca7eae0c-aa9a-4691-a2cb-96cf61984efc',  # R 01 | G 02 | B 5/5 | KH,PL+EM | COMPLETE
        'dca288e7-1437-4087-9f33-4815f1491273',  # Resolved_R 01 | G 02 | B 5/5 | KH,PL+EM (Andres)
    ),
    (
        '28ecb8e9-4828-482c-8f51-99f6cf3aa7cf',  # R 01 | G 03 | B 1/5 | AF,MP+IG,SL | COMPLETE
        'c34a0121-b6a5-4d3c-bebe-08ca0623535c',  # Resolved_R 01 | G 03 | B 1/5 | AF,MP+IG,SL (Andres)
    ),
    (
        '6ca6571e-d6f3-4450-89f5-a346681892ba',  # R 01 | G 03 | B 2/5 | MP+AR,CM/MIM/JB | COMPLETE
        '4e2e28ef-b866-4b56-8e24-5701638ccba6',  # Resolved_R 01 | G 03 | B 2/5 | MP+AR,CM/MIM/JB | (Lotte)
    ),
    (
        '3f9f1b6c-921a-4a57-82b2-08fe04651d15',  # R 01 | G 03 | B 3/5 | MP+AR,MC,EM | COMPLETE
        '6527b29e-0c54-48d6-9464-14d6068a3c57',  # Resolved_R 01 | G 03 | B 3/5 | MP+AR,MC,EM | (Andres)
    ),
    (
        'de889e2c-9366-47c6-8e6a-47f4c14b3619',  # R 01 | G 03 | B 4/5 | MP+EM,SL | COMPLETE
        '7e1e117d-0b88-4510-baf0-09eba15d18bf',  # Resolved_R 01 | G 03 | B 4/5 | MP+EM,SL (Maria-Inti)
    ),
    (
        '61777f54-b701-4a82-9362-16d217ea2bec',  # R 01 | G 03 | B 5/5 | MP+AR,CM | COMPLETE
        'dca31640-31d0-40c1-9a9c-9d87295f1e26',  # Resolved_R 01 | G 03 | B 5/5 | MP+AR,CM | (Lotte)
    ),
    (
        '08ae0534-1fec-4ffb-88a1-b889ac41d6b6',  # R 01 | G 04 | B 1/5 | AM,EM, AR | last batch
        '578f2341-1bb4-44d4-88e8-8e1f1180d80c',  # Resolved_R 01 | G 04 | B 1/5 | AM,EM, AR | (MI ongoing)
    ),
    (
        'd3bf790a-1639-4248-8304-10fe9a9fe92f',  # R 01 | G 04 | B 2/5 | ANS,AW,IP | COMPLETE
        'd83dfdff-6a65-4d67-b08b-2927dd16d3c0',  # Resolved_R 01 | G 04 | B 2/5 | ANS,AW,IP | (Lotte)
    ),
    (
        '0b902e2a-a7e8-4acf-9ab2-1f3b08db85a3',  # R 01 | G 04 | B 3/5 | ANS,IP+MP | COMPLETE
        '0778e8d6-b1b3-4804-872e-cb6825b4b448',  # Resolved_R 01 | G 04 | B 3/5 | ANS,IP+MP | (Lotte)
    ),
    (
        'eb958485-753e-4c75-ba1c-5a5a9f99cbd7',  # R 01 | G 04 | B 4/5 | AW,IP+EM, ANS | COMPLETE
        '309b77f0-555d-4fd3-b888-769877d503f5',  # Resolved_R 01 | G 04 | B 4/5 | AW,IP+EM, ANS | ( Am-resolved until 86)
    ),
    (
        'eb958485-753e-4c75-ba1c-5a5a9f99cbd7',  # R 01 | G 04 | B 4/5 | AW,IP+EM, ANS | COMPLETE
        'c535518a-7192-4b07-aa57-f086e76758fc',  # Resolved_R 01 | G 04 | B 4/5 | AW,IP+EM, ANS | (Andres & Maria-Inti)
    ),
    (
        'a1610cad-b408-4a4e-968b-831814a78747',  # R 01 | G 04 | B 5/5 | ANS,AW,IP | COMPLETE
        'e5145878-2769-4121-bfc4-0ea9507da932',  # Resolved_R 01 | G 04 | B 5/5 | ANS,AW,IP | (Andres)
    ),
    (
        'cd7e8d86-24ab-4f19-b3c0-48738bb1ffae',  # R 01 | G 05 | B 1/5 | JB,MC,PN | COMPLETE
        'b6249390-0400-4bd2-a15e-2519ca1046c0',  # Resolved_R 01 | G 05 | B 1/5 | JB,MC,PN (Andres)
    ),
    (
        '21923ddc-08ad-4110-917e-ec3e45b597f2',  # R 01 | G 05 | B 2/5 | JB,PN+MIM | COMPLETE
        'd264f8c4-b8fd-4049-9b00-17d368689119',  # Resolved_R 01 | G 05 | B 2/5 | JB,PN+MIM | (Lotte)
    ),
    (
        '00cf4c34-40df-45dd-90f1-24168d2f9bae',  # R 01 | G 05 | B 3/5 | JB,PN+EM | COMPLETE
        '8cfb1443-1926-40fc-892a-a87b2ac33309',  # Resolved_R 01 | G 05 | B 3/5 | JB,PN+EM (Andres)
    ),
    (
        'a4abe367-d9b2-47b6-921d-a8c66534ea7b',  # R 01 | G 05 | B 4/5 | JB,PN+EM | COMPLETE
        '2db195d3-52a5-4ae9-a4d9-ba3ca8be75de',  # Resolved_R 01 | G 05 | B 4/5 | JB,PN+EM | (Andres)
    ),
    (
        'a801a769-60d2-4e3c-9b72-39c61bd37f33',  # R 01 | G 05 | B 5/5 | JB,PN+AM | COMPLETE
        '66314b8a-b9d1-42cc-863f-2bacef0da9d2',  # Resolved_R 01 | G 05 | B 5/5 | JB,PN+AM (Lotte)
    ),
]

INOUT_SCOPE_IDS_RESOLVED = [li for (_, li) in INOUT_ANNOTATION_SCOPES]
INOUT_SCOPE_IDS_ANNOTATE = [li for (li, _) in INOUT_ANNOTATION_SCOPES]
INOUT_SCOPE_IDS = INOUT_SCOPE_IDS_RESOLVED + INOUT_SCOPE_IDS_ANNOTATE

TAXONOMY_SCOPE_IDS = ['706a1970-7400-4eab-a029-e2e214a09f39']
