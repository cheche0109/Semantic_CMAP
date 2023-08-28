from Bio.PDB import PDBParser
from mPdbSeqresIterator import mPdbSeqresIterator
#from mPDBData import AA3to1
from difflib import SequenceMatcher
from Bio import Align
from scipy.spatial.distance import pdist, squareform
import numpy as np
import seaborn as sns
import argparse
import os
import pandas as pd
from PIL import Image, ImageColor
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import logging

warnings.filterwarnings("ignore", category=PDBConstructionWarning)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s: %(message)s',
                    handlers=[logging.FileHandler(f"cmap_annotation_230516.log", 'w'), logging.StreamHandler()])

# import cv2
# #from google.colab.patches import cv2_imshow
# from itertools import groupby



AA3to1 = {'ALA':'A', 'CYS':'C', 'ASP':'D', 'GLU':'E', 'PHE':'F',
          'GLY':'G', 'HIS':'H', 'ILE':'I', 'LYS':'K', 'LEU':'L',
          'MET':'M', 'ASN':'N', 'PRO':'P', 'GLN':'Q', 'ARG':'R', 
          'SER':'S', 'THR':'T', 'VAL':'V', 'TRP':'W', 'TYR':'Y',
          'SEC':'U', 'PYL':'O', 'MSE':'M'}


def calc_residue_dist(residue_one, residue_two) :
    """Returns the C-alpha distance between two residues"""
    C_one = 'CB' if 'CB' in residue_one else 'CA'
    C_two = 'CB' if 'CB' in residue_two else 'CA'
    diff_vector  = residue_one[C_one].coord - residue_two[C_two].coord
    return np.sqrt(np.sum(diff_vector * diff_vector))


def path_check(path):
    """Check if the path exists, if not, create it"""
    if not os.path.exists(path):
        os.makedirs(path)


def label_color(classes):
    """Get the color of labels"""
    colorpalette = sns.color_palette("husl", len(classes)).as_hex()
    labels = dict()
    for i, l in enumerate(classes):
        logging.info(f"{l}: {i+1}, {colorpalette[i]}")
        labels[l] = [i+1, ImageColor.getcolor(colorpalette[i], "RGB")]

    return labels


def read_pdb(pdb_file):
    """Read pdb file"""
    pdbid = os.path.basename(pdb_file).split('.')[0]
    parser = PDBParser()
    structure = parser.get_structure(f"{pdbid}", pdb_file)

    return structure


def get_dbref(pdb_file):
    """Get the dbref from pdb file"""
    dbref = list()
    with open(pdb_file) as handle:
        for record in mPdbSeqresIterator(handle):
            row = dict()
            row['database'] = record[0].split(':')[0]
            row['id'] = record[0].split(':')[1]
            row['name'] = record[1].split(':')[1]
            row['seq_init'] = record[2].split(':')[1]
            row['seq_end'] = record[3].split(':')[1]
            row['dbxref_init'] = record[4].split(':')[1]
            row['dbxref_end'] = record[5].split(':')[1]
            dbref.append(row)

    return dbref
    

# def choose_chain(structure, pdb_interval, min_missing='global'):
#     """Choose the chain from pdb file"""
#     # check the name of chains
#     # mhc, tcr or antibody are not in the chain name
#     chain_list = []
#     for key, value in structure.header['compound'].items():
#         logging.debug(f"Chain: {value['chain']}, molecule: {value['molecule']}")
#         if not any([x in value['molecule'].upper() for x in ['MHC', 'TCR', 'T-CELL', 'IGE', 'ANTIBODY']]):
#             chain_list.extend(map(lambda x:x.upper(), value['chain'].split(', ')))
    
#     #logging.debug(f"PDB chain list: {chain_list}")
#     # check the missing residues
#     missing_residues = dict()
#     for i in structure.header['missing_residues']:
#         if i['chain'] in chain_list:
#             if i['chain'] not in missing_residues:
#                 missing_residues[i['chain']] = dict()
#             missing_residues[i['chain']][i['ssseq']] = AA3to1[i['res_name']]
    
#     # check the sequence numbering is within the pdb interval


#     # choose the chain with the least missing residues
#     if len(missing_residues) > 0:
#         if min_missing == 'global':
#             # choose the chain with the least missing residues in the whole protein
#             chain = min(missing_residues, key=lambda x:len(missing_residues[x]))
#         elif min_missing == 'center':
#             # choose the chain with the least missing residues in the center of the protein
#             # maxize the missing residues in the terminal regions
#             pass
#     else:
#         chain = chain_list[0]
#         missing_residues[chain] = dict()

#     logging.debug(f"Chain {chain} is chosen.")
    
#     return chain, missing_residues[chain]


# def pdb2seq_cmap(structure, chain_id):
#     """Get the sequence (with missing residues) from pdb file"""

#     # use structure[0], if a structure was determined by NMR, it would have multiple models
#     model = structure[0]
#     chain = model[chain_id]
#     seq = dict()
#     coords = []
#     for residue in chain:
#         het = residue.get_id()[0]
#         resseq = residue.get_id()[1]
#         if het == ' ':
#             try:
#                 seq[resseq] = AA3to1[residue.get_resname()]
#             except KeyError:
#                 seq[resseq] = 'X'
#                 print(f"{residue.get_resname()} is not a standard amino acid (23). Replaced by 'X'.")
        
#             # use C beta if exists, otherwise use C alpha (alanine)
#             carbon = 'CB' if 'CB' in residue else 'CA'
#             coords.append(residue[carbon].get_coord())
    
#     distance = pdist(coords)
#     cmap = squareform(distance)

#     return seq, cmap


def model2seq(structure, chain_id):

    structure_residues = dict()
    for residue in structure[0][chain_id]:
        het = residue.get_id()[0]
        resseq = residue.get_id()[1]
        if het == ' ':
            try:
                structure_residues[resseq] = AA3to1[residue.get_resname()]
            except KeyError:
                structure_residues[resseq] = 'X'
                print(f"{residue.get_resname()} is not a standard amino acid (23). Replaced by 'X'.")
    
    missing_residues = dict()
    for m in structure.header['missing_residues']:
        if m['chain'] == chain_id:
            missing_residues[m['ssseq']] = AA3to1[m['res_name']]
    
    for key, value in structure.header['compound'].items():
        if chain_id.lower() in value['chain'].split(', '):
            molecule_name = value['molecule']
            break
        else:
            molecule_name = None

    return structure_residues, missing_residues, molecule_name
    

def choose_chain(structure, pdb_interval, min_missing='global'):
    # use structure[0], if a structure was determined by NMR, it would have multiple models
    model = structure[0]
    chain_list = [i.id for i in model.get_list()]
    num_min_missing = 1000
    chain = None
    residues = []

    for c in chain_list:

        structure_residues, missing_residues, molecule_name = model2seq(structure, c)
        logging.debug(f"Chain {c}, molecule {molecule_name}")
        
        # check the molecule name
        if molecule_name is None or molecule_name.upper() in ['MHC', 'TCR', 'T-CELL', 'IGE', 'ANTIBODY']:
            continue
        else:
            tmp_residue = {**structure_residues, **missing_residues}
            complete_residues = {i:tmp_residue[i] for i in sorted(tmp_residue)}
        
        # check the sequence numbering is within the pdb interval
        residues_interval = [[sorted(complete_residues.keys())[0], sorted(complete_residues.keys())[-1]]]
        if is_interval_overlap(pdb_interval, residues_interval) < 0.97:
            logging.debug(f"Chain {c} starts at {residues_interval[0]}, ends at {residues_interval[-1]}.")
            logging.debug(f"PDB interval starts at {pdb_interval[0]}, ends at {pdb_interval[-1]}.")
            continue

        # choose the chain with the least missing residues
        if len(missing_residues) > 0:
            if min_missing == 'global':
                # choose the chain with the least missing residues in the whole protein
                num_missing = len(missing_residues)
                if num_missing < num_min_missing:
                    chain = c
                # chain = min(missing_residues, key=lambda x:len(missing_residues[x]))
            elif min_missing == 'center':
                # choose the chain with the least missing residues in the center of the protein
                # maxize the missing residues in the terminal regions
                pass
        else:
            chain = c

        if chain is None:
            logging.warning(f"No chain is chosen.")
            return None, None
        else:
            residues.append(complete_residues)
            residues.append(structure_residues)
            residues.append(missing_residues)
            logging.debug(f"Chain {chain} is chosen.")
        return chain, residues


def model2cmap(structure, chain_id, structure_residues):
    
    coords = []
    for idx, residue in structure_residues.items():
        carbon = 'CB' if 'CB' in residue else 'CA'
        coords.append(structure[0][chain_id][idx][carbon].get_coord())
    distance = pdist(coords)
    cmap = squareform(distance)

    return cmap


def insert_missing(cmap, structure_residue, missing_residue):
    """Insert missing residues into cmap"""
    tmp_residue = {**structure_residue, **missing_residue}
    complete_residue = {i:tmp_residue[i] for i in sorted(tmp_residue)}
    for key, value in missing_residue.items():
        # locate the position of the missing residue in the complete residue
        pos = list(complete_residue.keys()).index(key)
        
        # insert the row and column of the missing residue
        cmap = np.insert(cmap, pos, np.nan, axis=0)
        cmap = np.insert(cmap, pos, np.nan, axis=1)
    
    return complete_residue, cmap


def pdb2cmap(structure, chain, structure_residues, missing_residues):
    
    cmap_origin = model2cmap(structure, chain, structure_residues)
    # insert missing residues into cmap
    _, cmap = insert_missing(cmap_origin, structure_residues, missing_residues)

    return cmap

    


def is_interval_overlap(domain_interval, dbxref_interval):
    
    interval1_length = 0
    overlap_length = 0

    for interval in domain_interval:

        # calculate the start and end points of the overlap between the intervals
        overlap_start = max(domain_interval[0][0], dbxref_interval[0][0])
        overlap_end = min(domain_interval[-1][1], dbxref_interval[-1][1])

        # calculate the length of the overlap and the length of the smaller interval
        overlap_length += max(0, overlap_end - overlap_start)
        #min_interval_length = min(interval[1] - interval[0], interval2[1] - interval2[0])
        interval1_length += (domain_interval[-1][1] - domain_interval[0][0])

    # calculate the ratio of the overlap length to the smaller interval length
    overlap_ratio = overlap_length / interval1_length

    return overlap_ratio


def check_identity(seq1, seq2):
    """Check if two sequences are identical"""
    

    if seq1 == seq2:
        return 1
    else:
        #return 0
        # quick check
        aligner = Align.PairwiseAligner()
        # aligner.mode = 'local'
        # aligner.match_score = 1
        # aligner.mismatch_score = -1
        # aligner.open_gap_score = -1
        # aligner.extend_gap_score = -1
        # aligner.target_end_gap_score = 0
        # aligner.query_end_gap_score = 0
        # aligner.query_internal_extend_gap_score = 0
        # aligner.target_internal_extend_gap_score = 0
        # aligner.query_internal_open_gap_score = 0
        # aligner.target_internal_open_gap_score = 0
        # aligner.query_internal_end_gap_score = 0

        alignment = aligner.align(seq1, seq2)

        if len(seq1) != len(seq2):
            logging.warning(f"The length of two sequences are not equal, {len(seq1)} vs {len(seq2)}.")
        
        return alignment[0].score / len(seq1)

        #return ratio



def str2array(string):
    """Convert string to numpy array"""
    return np.asarray(eval(string))#.flatten()


def annotate_cmap(cmap, thre_dist, domain, seq, domain_interval, pdb_residue, pdb_interval, dbxref_interval, labels):
    """Annotate cmap to generate the groud truth (array 0-num(class)) and color map (rgb array)"""

    cmap_origin = np.where(cmap < thre_dist, 0, 255)
    # check sequence identity
    # interval_seq = seq[dbxref_interval[0]-1:dbxref_interval[1]]
    # pdb_full_seq = ''.join(list(pdb_residue.values()))
    # pdb_init = list(pdb_residue.keys())[0]
    # pdb_interval_seq = pdb_full_seq[pdb_interval[0]-pdb_init-1:pdb_interval[1]-pdb_init]
    

    domain_seq = seq[(domain_interval[0][0] - 1):domain_interval[-1][1]]
    pdb_full_seq = ''.join(list(pdb_residue.values()))
    pdb_init = list(pdb_residue.keys())[0]
    # interpro domain position - (shift between pdb and uniprot)
    pdb_domain_loc = domain_interval - dbxref_interval[0][0] + pdb_interval[0][0]
    
    domain_annot_start = max(pdb_domain_loc[0][0] - pdb_init, 0)
    domain_annot_end = pdb_domain_loc[-1][1] - pdb_init + 1
    pdb_domain_seq = pdb_full_seq[domain_annot_start:domain_annot_end]
    
    logging.debug(f"UniPSeq: {seq}")
    logging.debug(f"StruSeq: {pdb_full_seq}")

    logging.debug(f"Corrected Domain interval: {pdb_domain_loc}")
    logging.debug(f"Corrected Domain Seqindex: {domain_annot_start} - {domain_annot_end}")
    logging.debug(f"DomSeq: {domain_seq}")
    logging.debug(f"PDBSeq: {pdb_domain_seq}")
    ratio = check_identity(domain_seq, pdb_domain_seq)

    logging.info(f"Domain sequence identity {ratio}")
    if  ratio > 0.9:
        logging.info("Sequence identity check passed.")
    else:
        logging.info("Sequence identity check failed.")
        return (None, None, None)

    
    # annotate cmap
    cmap_gt = np.zeros((cmap.shape[0], cmap.shape[1]), dtype=int)
    cmap_color = np.zeros((cmap.shape[0], cmap.shape[1], 3), dtype=int)
    
    for interval in pdb_domain_loc:
        domain_annot_start = max(interval[0] - pdb_init, 0)
        domain_annot_end = interval[1] - pdb_init + 1

        mask = cmap[domain_annot_start:domain_annot_end, domain_annot_start:domain_annot_end] < thre_dist
    
        #print(cmap[domain_annot_start, domain_annot_end])
        cmap_gt[domain_annot_start:domain_annot_end, domain_annot_start:domain_annot_end][mask] = labels[domain][0]
        
        cmap_color[domain_annot_start:domain_annot_end, domain_annot_start:domain_annot_end][mask] = labels[domain][1]

    return (cmap_origin, cmap_gt, cmap_color)
    


def main(args):

    pdb_dir = args.Input_pdb
    annot_file = args.Input_annot

    cmap_dir = args.Original_cmap
    gt_dir = args.Ground_truth
    color_dir = args.Mask_color

    thre_1 = args.Threshold_short
    thre_2 = args.Threshold_long
    thre_dist = args.Threshold_distance
    
    # check output path
    path_check(cmap_dir)
    path_check(gt_dir)
    path_check(color_dir)

    # read annotation file
    df_annot = pd.read_csv(annot_file, index_col=0)

    # annotation classes
    classes = df_annot['domain.name'].unique()
    labels = label_color(classes)

    # ignore missing residues
    for idx, row in df_annot.iterrows():
        
        pdb = row['pdb_y']
        pdb_file = os.path.join(pdb_dir, f"{pdb}.pdb")
        print()
        domain = row['domain.name']
        logging.info(f"Processing {pdb_file}")
        logging.info(f"Annotating {pdb} {domain}")

        # check sequence identity between pdb and fasta dbxref regions
        # read pdb fasta file including missing residues
        # use domain_init - dbxref_init + 1 as the start position for pdb fasta file including missing residues
        # find positions of the pdb fasta excluding missing residues in cmap
        # annotate the cmap
        domain_interval = str2array(row['domain.intervals'])
        dbxref_interval = str2array(row['dbxrefs.interval'])
        pdb_interval = str2array(row['seq.interval'])
        coverage = is_interval_overlap(domain_interval, dbxref_interval)
        logging.debug(f"Interval coverage: {coverage}")
        if not coverage > 0.97:
            continue

        
        logging.debug(f"Domain interval: {domain_interval}")
        logging.debug(f"DBXREF interval: {dbxref_interval}")
        logging.debug(f"PDB--- interval: {pdb_interval}")
        
        #dbref = get_dbref(pdb_file)
        structure = read_pdb(pdb_file)

        chain, residues = choose_chain(structure, pdb_interval, min_missing='global')
        logging.info(f"Resolution {row['resolution']}")
        logging.info(f"Chain {chain}, complete residue {len(residues[0])}, structure residue {len(residues[1])}, missing residue {len(residues[2])}")
        cmap = pdb2cmap(structure, chain, residues[1], residues[2])

        #print(f"Labels: {labels}")
        cmaps = annotate_cmap(cmap, thre_dist, domain, row['sequence'], domain_interval, residues[0], pdb_interval, dbxref_interval, labels)
        if cmaps[0] is None:
            continue

        # save cmap
        #print(cmaps[0].shape)
        cmap_origin_file = os.path.join(cmap_dir, f"{pdb}_{chain}_{labels[domain][0]}.png")
        Image.fromarray(cmaps[0].astype(np.uint8), mode='L').save(cmap_origin_file)
 
        #print(cmaps[1].shape)
        cmap_gt_file = os.path.join(gt_dir, f"{pdb}_{chain}_{labels[domain][0]}.png")
        Image.fromarray(cmaps[1].astype(np.uint8), mode='L').save(cmap_gt_file)
        
        #print(cmaps[2].shape)
        cmap_color_file = os.path.join(color_dir, f"{pdb}_{chain}_{labels[domain][0]}.png")
        Image.fromarray(cmaps[2].astype(np.uint8)).save(cmap_color_file)

        logging.info(f"Annotation for {pdb} {chain} {domain} is done.")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Annotation for domain on contact map of proteins')
    parser.add_argument('-i', action='store', dest='Input_pdb', type=str, required=True, help='Input pdb dir')
    parser.add_argument('-a', action='store', dest='Input_annot', type=str, required=True, help='Input annotation csv file')
    parser.add_argument('-T1', action='store', dest='Threshold_short', type=int, default=50, help='shortest length of CMAP (default: 50)')
    parser.add_argument('-T2', action='store', dest='Threshold_long', type=int, default=800, help='longest length of CMAP (default: 800)')
    parser.add_argument('-CMAP', action='store', dest='Original_cmap', type=str, required=True, help='Output original cmap dir')
    parser.add_argument('-GT', action='store', dest='Ground_truth', type=str, required=True, help='Output ground Truth dir')
    parser.add_argument('-Color', action='store', dest='Mask_color', type=str, required=True, help='Output mask color dir')
    parser.add_argument('-Dist', action='store', dest='Threshold_distance', type=int, default=8, help='Angstrom threshold (default: 8)')
    parser.add_argument('-v', action='version', version='%(prog)s 0.1', help='Show version')

    args = parser.parse_args()


    main(args)