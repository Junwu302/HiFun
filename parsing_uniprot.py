# -*- coding: utf-8 -*-
"""
@File   : parsing_uniprot.py.py    
@Contact: junwu302@gmail.com
@Usage  : xxxxxx
@Time   : 2022/9/8 16:10 
@Version: 1.0
@Desciption: Parsing the file downloaded from https://www.uniprot.org/help/downloads
"""
# copy from deepGOplus package (uni2pandas.py)
import logging
import click as ck
import pandas as pd
import gzip
from parsing_GO import Ontology, go_propagate

EXP_CODES = set([
    'EXP', 'IDA', 'IPI', 'IMP', 'IGI', 'IEP', 'TAS', 'IC',
    'HTP', 'HDA', 'HMP', 'HGI', 'HEP'])
# CAFA4 Targets
CAFA_TARGETS = set([
    '287', '3702', '4577', '6239', '7227', '7955', '9606', '9823', '10090',
    '10116', '44689', '83333', '99287', '226900', '243273', '284812', '559292'])
logging.basicConfig(level=logging.INFO)
@ck.command()
@ck.option(
    '--go-file', '-gf', default='db/go-basic.obo',
    help='Gene Ontology file in OBO Format')
@ck.option(
    '--swissprot-file', '-sf', default='db/uniprot_sprot.dat.gz',
    help='UniProt/SwissProt knowledgebase file in text format (archived)')
@ck.option(
    '--out-file', '-o', default='db/swissprot.pkl',
    help='Result file with a list of proteins, sequences and annotations')

def is_cafa_target(org): # taxonomy ID
    return org in CAFA_TARGETS

def is_exp_code(code):
    return code in EXP_CODES

def experimental_filtering(swissprot_df):
    logging.info('Filtering proteins with experimental annotations')
    index = []
    annotations = []
    for i, row in enumerate(swissprot_df.itertuples()):
        annots = []
        for annot in row.annotations:
            go_id, code = annot.split('|')
            if is_exp_code(code):
                annots.append(go_id)
        # Ignore proteins without experimental annotations
        if len(annots) == 0:
            continue
        index.append(i)
        annotations.append(annots)
    swissprot_df = swissprot_df.iloc[index]
    swissprot_df = swissprot_df.reset_index()
    swissprot_df['exp_annotations'] = annotations
    return swissprot_df

def main(swissprot_file='db/uniprot_sprot.dat.gz', go_file = 'db/go-basic.obo', out_file = 'db/swissprot.pkl'):
    proteins = list()
    accessions = list()
    sequences = list()
    annotations = list()
    interpros = list()
    orgs = list()
    with gzip.open(swissprot_file, 'rt') as f:
        prot_id = ''
        prot_ac = ''
        seq = ''
        org = ''
        annots = list()
        ipros = list()
        for line in f:
            items = line.strip().split('   ')
            if items[0] == 'ID' and len(items) > 1:
                if prot_id != '':
                    proteins.append(prot_id)
                    accessions.append(prot_ac)
                    sequences.append(seq)
                    annotations.append(annots)
                    interpros.append(ipros)
                    orgs.append(org)
                prot_id = items[1]
                annots = list()
                ipros = list()
                seq = ''
            elif items[0] == 'AC' and len(items) > 1:
                prot_ac = items[1]
            elif items[0] == 'OX' and len(items) > 1:
                if items[1].startswith('NCBI_TaxID='):
                    org = items[1][11:]
                    end = org.find(' ')
                    org = org[:end]
                else:
                    org = ''
            elif items[0] == 'DR' and len(items) > 1:
                items = items[1].split('; ')
                if items[0] == 'GO':
                    go_id = items[1]
                    code = items[3].split(':')[0]
                    annots.append(go_id + '|' + code)
                if items[0] == 'InterPro':
                    ipro_id = items[1]
                    ipros.append(ipro_id)
            elif items[0] == 'SQ':
                seq = next(f).strip().replace(' ', '')
                while True:
                    sq = next(f).strip().replace(' ', '')
                    if sq == '//':
                        break
                    else:
                        seq += sq

        proteins.append(prot_id)
        accessions.append(prot_ac)
        sequences.append(seq)
        annotations.append(annots)
        interpros.append(ipros)
        orgs.append(org)
    swissprot_df = pd.DataFrame({
        'proteins': proteins,
        'accessions': accessions,
        'sequences': sequences,
        'annotations': annotations,
        'interpros': interpros,
        'orgs': orgs
    })

    swissprot_df = experimental_filtering(swissprot_df)
    swissprot_df['prop_annotations'] = go_propagate(go = Ontology(go_file), annot_list= swissprot_df['exp_annotations'])

    cafa_target = []
    for i, row in enumerate(swissprot_df.itertuples()):
        if is_cafa_target(row.orgs):
            cafa_target.append(True)
        else:
            cafa_target.append(False)
    swissprot_df['cafa_target'] = cafa_target
    swissprot_df.to_pickle(out_file)
    logging.info('Successfully saved %d proteins' % (len(swissprot_df),))

if __name__ == '__main__':
    main()
