#!/usr/bin/env python
# coding: utf-8
# %%

# %%


import os
import pandas as pd
import time
import concurrent.futures

from rdflib import Graph

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

df = pd.read_csv('core/regulatory_map_fin.csv')

mapping_dct = {}

for index, row in df.iterrows():
    ttl_file = row['file_ttl']
    regulatory = row['regulatory']
    if ttl_file in mapping_dct:
        mapping_dct[ttl_file].append(regulatory)
    else:
        mapping_dct[ttl_file] = [regulatory]

def extract_turtle(regulatory_id, turtle_file_path, text_file_path):
    target_file = text_file_path.replace('new_1_text_files', 'new_2_turtle_files')
    target_file = target_file.replace('txt', 'ttl')
    
    print(f'{regulatory_id} {turtle_file_path} {target_file}')
    
    if os.path.exists(target_file):
        print(f'{target_file} already exists. Skipping...')
        with open('result2.txt', 'a') as f: 
            f.write(f'{regulatory_id}\n')
        return
    
    start = time.time()
    
    try:
        g = Graph()
        g.parse(turtle_file_path, format='ttl')

        query = f"""
        prefix xsd: <http://www.w3.org/2001/XMLSchema#>
        prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        prefix dbo: <http://dbpedia.org/ontology/>
        prefix dct: <http://purl.org/dc/terms/>
        prefix owl: <http://www.w3.org/2002/07/owl#>
        prefix wd: <https://www.wikidata.org/wiki/>
        prefix lexid-s: <https://w3id.org/lex-id/schema/>
        prefix lexid: <https://w3id.org/lex-id/data/>

        construct {{
            ?z ?p ?o .
            ?o ?p1 ?o2 .
        }}
        where {{
            {{
                select distinct ?z ?p ?o
                where {{
                    lexid:{regulatory_id} (
                        lexid-s:considers 
                        | dct:description 
                        | dct:description 
                        | lexid-s:name 
                        | lexid-s:hasCreator 
                        | lexid-s:hasDictum 
                        | lexid-s:hasCreator 
                        | lexid-s:hasEnactionDate 
                        | lexid-s:hasEnactionLocation 
                        | lexid-s:hasEnactionOffice 
                        | lexid-s:hasEnactionOfficial 
                        | lexid-s:hasPromulgationDate 
                        | lexid-s:hasPromulgationLocation 
                        | lexid-s:hasPromulgationOffice 
                        | lexid-s:hasPromulgationOfficial 
                        | lexid-s:hasPromulgationPlace 
                        | lexid-s:hasRegulationNumber 
                        | lexid-s:hasRegulationYear 
                        | rdfs:label 
                        | rdf:type
                        | lexid-s:hasLegalBasis 
                        | lexid-s:LegalBasisOf 
                        | lexid-s:hasContent 
                        | lexid-s:isContentOf 
                        | lexid-s:hasPart 
                        | lexid-s:isPartOf 
                    )* ?z .
                    ?z ?p ?o .
                    FILTER (?p NOT IN (rdf:type
                                        , owl:sameAs
                                        , lexid-s:adds
                                        , lexid-s:hasAdditionContent
                                        , lexid-s:hasAdditionTarget
                                        , lexid-s:deletes
                                        , lexid-s:modifies
                                        , lexid-s:hasModificationContent
                                        , lexid-s:hasModificationTarget
                                        , lexid-s:hasAct
                                        , lexid-s:hasActType
                                        , lexid-s:hasElement
                                        , lexid-s:hasCondition
                                        , lexid-s:hasModality
                                        , lexid-s:hasObject
                                        , lexid-s:hasQualifier
                                        , lexid-s:hasQualifierType
                                        , lexid-s:hasQualifierValue
                                        , lexid-s:hasRule
                                        , lexid-s:hasSubject
                                        , lexid-s:isRuleOf
                                        , lexid-s:refersTo
                                        , lexid-s:amendedBy
                                        , lexid-s:amends
                                        , lexid-s:implementedBy
                                        , lexid-s:implements
                                        , lexid-s:repealedBy
                                        , lexid-s:repeals
                                        , lexid-s:undefined))
                }}
            }}
            OPTIONAL {{
                ?o ?p1 ?o2 .
                FILTER (?p1 IN (rdf:type) && ?o2 NOT IN (owl:Thing
                                                        , lexid-s:LawAmandment
                                                        , lexid-s:Act
                                                        , lexid-s:AgencyRegulation
                                                        , lexid-s:AmendmentToTheConstitution
                                                        , lexid-s:CityOrdinance
                                                        , lexid-s:Constitution
                                                        , lexid-s:GovernmentRegulation
                                                        , lexid-s:GovernmentRegulationInLieuOfAct
                                                        , lexid-s:GovernorRegulation
                                                        , lexid-s:JointRegulation
                                                        , lexid-s:MayorRegulation
                                                        , lexid-s:MinisterialDecree
                                                        , lexid-s:MinisterialRegulation
                                                        , lexid-s:PeoplesConsultativeAssemblyResolution
                                                        , lexid-s:PresidentialDecree
                                                        , lexid-s:PresidentialRegulation
                                                        , lexid-s:ProvincialOrdinance
                                                        , lexid-s:RegencyOrdinance
                                                        , lexid-s:RegentRegulation
                                                        , lexid-s:LegalDocumentContent
                                                        , lexid-s:RuleExpression))
            }}
        }}
        """

        g.bind('rdf', 'http://www.w3.org/1999/02/22-rdf-syntax-ns#')
        g.bind('lexid-s', 'https://w3id.org/lex-id/schema/')
        g.bind('lexid', 'https://w3id.org/lex-id/data/')
        g.bind('dct', 'http://purl.org/dc/terms/')
        g.bind('xsd', 'http://www.w3.org/2001/XMLSchema#')
        g.bind('rdfs', 'http://www.w3.org/2000/01/rdf-schema#')
        g.bind('dbo', 'http://dbpedia.org/ontology/')
        g.bind('owl', 'http://www.w3.org/2002/07/owl#')
        g.bind('wd', 'https://www.wikidata.org/wiki/')

        results = g.query(query)

        f = open(target_file, 'w')

        prefixes = ['@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .',
                    '@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .',
                    '@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .',
                    '@prefix dbo: <http://dbpedia.org/ontology/> .',
                    '@prefix dct: <http://purl.org/dc/terms/> .',
                    '@prefix owl: <http://www.w3.org/2002/07/owl#> .',
                    '@prefix wd: <https://www.wikidata.org/wiki/> .',
                    '@prefix lexid-s: <https://w3id.org/lex-id/schema/> .',
                    '@prefix lexid: <https://w3id.org/lex-id/data/> .']

        for prefix in prefixes:
            res = F"{prefix}\n"
            f.write(res)

        f.write('\n')

        for result in results:
            subject = result[0].n3(g.namespace_manager)
            predicate = result[1].n3(g.namespace_manager)
            obj = result[2].n3(g.namespace_manager)
            res = F"{subject} {predicate} {obj} .\n"
            f.write(res)

        f.close()
        with open('result2.txt', 'a') as f: 
            f.write(f'{regulatory_id}\n')
        
    except Exception as e:
        print(e)
        with open('error2.txt', 'a') as f: 
            f.write(f'{regulatory_id} {turtle_file_path}\n')

    end = time.time()
    print(f'{regulatory_id} from {turtle_file_path} to {target_file} execution time {end - start} seconds')
    
    
def process_row(row):
    regulatory_id = row['regulatory']
    turtle_file_path = row['file_ttl']
    text_file_path = row['file_txt']
    extract_turtle(regulatory_id, turtle_file_path, text_file_path)
    

num_workers = 5

# ttl = 'turtle_files/ln/2000/uu33-2000.ttl'

# tmp = df[df['ttl_file'] == ttl]
# with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
#     futures = [executor.submit(process_row, row) for index, row in tmp.iterrows()]
#     concurrent.futures.wait(futures)

# print(f'All tasks completed for {ttl}')

ttls = reversed(list(mapping_dct.keys()))

for ttl in ttls:
    tmp = df[df['file_ttl'] == ttl]
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_row, row) for index, row in tmp.iterrows()]
        concurrent.futures.wait(futures)

    print(f'All tasks completed for {ttl}')


# %%
