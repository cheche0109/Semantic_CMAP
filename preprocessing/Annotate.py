
from Bio.PDB import *
import numpy as np
import matplotlib.pyplot as plt
#import numpy as np
import seaborn as sns
import argparse
import os, re
import pandas as pd
from PIL import Image
import cv2
#from google.colab.patches import cv2_imshow
from itertools import groupby

parser = argparse.ArgumentParser(description='Auto annotation for semantic segmantation')
parser.add_argument('-i', action='store', dest='Input_pdb', type=str, required=True, nargs='+', help='Input pdb file(s)')
parser.add_argument('-T1', action='store', dest='Threshold_short', type=int, default=50, help='shortest length of CMAP (default: 50)')
parser.add_argument('-T2', action='store', dest='Threshold_long', type=int, default=800, help='longest length of CMAP (default: 800)')
parser.add_argument('-CMAP', action='store', dest='Original_cmap', type=str, required=True, help='Original cmap dir')
parser.add_argument('-GT', action='store', dest='Ground_truth', type=str, required=True, help='Ground Truth dir')
parser.add_argument('-Color', action='store', dest='Mask_color', type=str, required=True, help='Mask color dir')
parser.add_argument('-Dist', action='store', dest='Threshold_distance', type=int, default=8, help='Angstrom threshold (default: 8)')

args = parser.parse_args()
pdb_dir = args.Input_pdb
cmap_dir = args.Original_cmap
gt_dir = args.Ground_truth
color_dir = args.Mask_color
thre_1 = args.Threshold_short
thre_2 = args.Threshold_long
thre_dist = args.Threshold_distance


def calc_residue_dist(residue_one, residue_two) :
    """Returns the C-alpha distance between two residues"""
    C_one = 'CB' if 'CB' in residue_one else 'CA'
    C_two = 'CB' if 'CB' in residue_two else 'CA'
    diff_vector  = residue_one[C_one].coord - residue_two[C_two].coord
    return np.sqrt(np.sum(diff_vector * diff_vector))

def aa_residues(chain):
    aa_names = standard_aa_names + ['SEC', 'PYL', 'MSE']
    All=list()
    AA_only = []
    for i in chain:
        All.append(i)
        if i.get_resname() in aa_names:
            AA_only.append(i)
    return AA_only, All

Convert_AA = {'ALA': 'A',
 'CYS':'C',
 'ASP':'D',
 'GLU':'E',
 'PHE':'F',
 'GLY':'G',
 'HIS':'H',
 'ILE':'I',
 'LYS':'K',
 'LEU':'L',
 'MET':'M',
 'ASN':'N',
 'PRO':'P',
 'GLN':'Q',
 'ARG':'R',
 'SER':'S',
 'THR':'T',
 'VAL':'V',
 'TRP':'W',
 'TYR':'Y',
 'SEC':'U', 
 'PYL':'O', 
 'MSE':'M'}

if __name__ == "__main__":

    if not os.path.exists(cmap_dir):
        os.makedirs(cmap_dir)
    if not os.path.exists(gt_dir):
        os.makedirs(gt_dir)
    if not os.path.exists(color_dir):
        os.makedirs(color_dir)

    for pdb_file in sorted(pdb_dir):
        #print(pdb_filename)

        pdb_filename = pdb_file.split('/')[-1]
        pdb_code = pdb_filename.split('.')[0]

        missing_position=list()
        missing_part=list()

        first_last=list()
        helix=list()
        sheet=list()
        sheet_sense=list()

        with open(pdb_file) as F:
            print(pdb_code)
            
            for line in F:

                res_miss = re.search(r'^REMARK\s+465\s+\w+\s+\w+\s+(\d{1,})',line)
                res_first_last = re.search(r'^DBREF',line)
                res_helix = re.search(r'^HELIX',line)
                res_sheet = re.search(r'^SHEET',line)

                if res_miss:
                    missing_position.append(int(res_miss.group(1)))

                if res_first_last:
                    first_last.append([int(float(line[14:18])),int(float(line[20:24]))])

                if res_helix:
                    initICode = line[25]
                    endICode = line[37]
                    if initICode == ' ' and endICode == ' ':
                        helix.append([int(line[21:25]),int(line[33:37])])
                    else:
                        f = open("Insert_helix.txt", "a")
                        f.write(pdb_filename+'_' + chain_id + '\n')
                        f.close()

                if res_sheet:
                    initICode = line[26]
                    endICode = line[37]
                    if initICode == ' ' and endICode == ' ':
                        sheet.append([int(line[22:26]), int(line[33:37])])
                        sheet_sense.append(int(line[38:40]))
                    else:
                        f = open("Insert_sheet.txt", "a")
                        f.write(pdb_filename+'_' + chain_id + '\n')
                        f.close()

        existing=list()
        # find the residues that do really exist
        for pos in range(int(first_last[0][0]), int(first_last[0][1])+1):
            if pos not in missing_position:
                existing.append(pos)

        #print(existing)
        start_aa = existing[0]
        new_existing = [x - int(start_aa) for x in existing]

        """ group the continuous residues """
        existing_part=list()
        fun = lambda x: x[1]-x[0]
        for k, g in groupby(enumerate(new_existing), fun):
            l1 = [j for i, j in g]
            existing_part.append(l1)


        """ connect the lists in the list """
        for i in range(len(existing_part)-1):
            existing_part[i+1][0] = existing_part[i][-1]+1
            for j in range(1,len(existing_part[i+1])):
                existing_part[i+1][j] = existing_part[i+1][j-1]+1

        """ find the longest fragment in the seq and thus generate the cutted cmap for the fragment """
        longest = existing_part[sorted([(i,len(l)) for i,l in enumerate(existing_part)], key=lambda t: t[1])[-1][0]]        

        
        new_existing = [item for sublist in existing_part for item in sublist]        


        change_pos = list()

        for idx in range(len(new_existing)):
            change_pos.append([existing[idx], new_existing[idx]])

        new_helix=list()
        new_sheet=list()

        new_start=0
        new_end=0


        for old_pair in helix:
            for name_pair in change_pos:
                if old_pair[0] == name_pair[0]:
                    new_start = name_pair[1]
                if old_pair[1] == name_pair[0]:
                    new_end = name_pair[1]
            new_helix.append([new_start, new_end])
        #print('new_helix:', new_helix)


        for old_pair in sheet:
            for name_pair in change_pos:
                if old_pair[0] == name_pair[0]:
                    new_start = name_pair[1]
                if old_pair[1] == name_pair[0]:
                    new_end = name_pair[1]
            new_sheet.append([new_start, new_end])


        #### Auto-annotation for segmentation: parallel & anti-parallel on off-diagnol
        anti_sheet_idx = list()
        parallel_sheet_idx=list()

        for idx in range(len(sheet_sense)):
            sense = sheet_sense[idx]
            if sense == -1:
            # anti_parallel
                anti_sheet_idx.append([idx-1, idx])
            elif sense == 1:
            # parallel  
                parallel_sheet_idx.append([idx-1, idx])
            else:
                pass

        parser = PDBParser(QUIET=False)
        structure = parser.get_structure(pdb_code, pdb_file)
        model = structure[0]

        chain_list = [i.get_id() for i in model.get_chains()]

        for chain_id in chain_list:
            aa, All= aa_residues(model[chain_id])
            """ get the position of the first aa and the last aa in the seq """

            if len(aa) != 0: 
                """ exclude the unreal chain like DNA"""

                if len(aa) == len(new_existing):

                    pos_aa_0 = aa[0].get_full_id()[3][1]
                    pos_aa_1 = aa[-1].get_full_id()[3][1]
                    """  exclue the chain contains the aa not from aa_names"""
                    with open(pdb_file) as F:
                        atom_list_0 = list()
                        atom_list_1 = list()
                        for line in F:
                            atom_info = re.search(r'^ATOM',line)
                            if atom_info:
                              if int(line[22:26]) == pos_aa_0:
                                atom_list_0.append(line[12:16].strip())
                              if int(line[22:26]) == pos_aa_1:
                                atom_list_1.append(line[12:16].strip())

                    if 'CA' not in atom_list_0 and 'CA' in atom_list_1:
                        f = open("_CA.txt", "a")
                        f.write(pdb_code + '_' + chain_id+'\n')
                        f.close()

                        aa = aa[1:]
                        if len(new_helix) > 0:
                            new_helix = [[x - 1, y - 1] for x, y in new_helix]
                            if new_helix[0][0] == -1:
                                new_helix[0][0] == 0
                        if len(new_sheet) > 0:
                            new_sheet = [[x - 1, y - 1] for x, y in new_sheet]
                            if new_sheet[0][0] == -1:
                                new_sheet[0][0] == 0                            

                        if longest[0] == new_existing[0]:
                            longest = longest[:-1]
                        else:
                            longest = longest[1:-1]
                    if 'CA' in atom_list_0 and 'CA' not in atom_list_1:
                        f = open("CA_.txt", "a")
                        f.write(pdb_code + '_' + chain_id+'\n')
                        f.close()
                        aa = aa[:-1]
                        if len(new_helix) > 0:
                            if new_helix[-1][-1] == new_existing[-1]:
                                new_helix[-1][-1] -= 1
                        if len(new_sheet) > 0:
                            if new_sheet[-1][-1] == new_existing[-1]:
                                new_sheet[-1][-1] -= 1   

                        if longest[-1] == new_existing[-1]:
                            longest = longest[:-1]
                        else:
                            longest = longest
                    if 'CA' not in atom_list_0 and 'CA' not in atom_list_1:
                        f = open("_CA_.txt", "a")
                        f.write(pdb_code + '_' + chain_id+'\n')
                        f.close()
                        aa = aa[1:-1]
                        if len(new_helix) > 0:
                            new_helix = [[x - 1, y - 1] for x, y in new_helix]
                            if new_helix[0][0] == -1:
                                new_helix[0][0] == 0
                            if new_helix[-1][-1] == new_existing[-1]:
                                new_helix[-1][-1] -= 1 
                        if len(new_sheet) > 0:
                            new_sheet = [[x - 1, y - 1] for x, y in new_sheet]
                            if new_sheet[0][0] == -1:
                                new_sheet[0][0] == 0
                            if new_sheet[-1][-1] == new_existing[-1]:
                                new_sheet[-1][-1] -= 1

                        if longest[0] == new_existing[0]:
                            if longest[-1] == new_existing[-1]:
                                longest = longest[:-2]
                            else:
                                longest = longest[:-1]
                        else:
                            if longest[-1] == new_existing[-1]:
                                longest = longest[:-2]
                                longest.insert(0, longest[0]-1)
                            else:
                                longest = longest[1:-1]


                    print(pdb_filename)
                    answer = np.zeros((len(aa), len(aa)), np.float)
                    mask = np.zeros((len(aa), len(aa)), np.float)
                    for row, residue_one in enumerate(aa):   
                        for col, residue_two in enumerate(aa) :
                          #print(row, residue_one, col, residue_two)
                            answer[row, col] = calc_residue_dist(residue_one, residue_two)
                          #print(answer)
                            mask[row,col] = 0

                    print(answer)
                    heat_map = np.where(answer < thre_dist, 0, 255)

                    cutted_cmap = heat_map[longest[0]:longest[-1], longest[0]:longest[-1]]


                    if thre_1 <= cutted_cmap[0].size <= thre_2:

                        """ Write the seq"""
                        seq = ''
                        for i in range(longest[0], longest[-1]+1):
                          AA_1 = Convert_AA[aa[i].resname]
                          seq += AA_1

                        f = open("Dataset_SC.fsa", "a")
                        f.write('>' + pdb_code + '_' + chain_id+ '_' + str(len(longest)) + '\n')
                        f.write(seq+'\n')
                        f.close()

                        """ Generate CMAP """
                        #cutted_cmap = heat_map
                        draw_heat_map = np.array(cutted_cmap, dtype='uint8')
                        print(draw_heat_map)
                        exit()
                        draw_heat_map = Image.fromarray(draw_heat_map)
                        draw_heat_map.save(cmap_dir + '/' + pdb_code + '_' + chain_id + '_origin.png')

                        
                        ##### Diagnol
                        helix_list = list() 
                        for s_e in new_helix:
                            for x in range(s_e[0], s_e[1]+1):
                                for y in range(s_e[0], s_e[1]+1):
                                    if heat_map[x][y] == 0:
                                        helix_list.append([x,y])

                        sheet_list = list()

                        for s_e in new_sheet:
                            for x in range(s_e[0], s_e[1]+1):
                                for y in range(s_e[0], s_e[1]+1):
                                    if heat_map[x][y] == 0:
                                        sheet_list.append([x,y])


                        ##### off-Diagnol
                        anti_parallel=list()
                        parallel=list()

                        if len(new_sheet) > 1:
                            for pair_sheet in anti_sheet_idx:
                                for x in range(new_sheet[pair_sheet[0]][0], new_sheet[pair_sheet[0]][1]+1):
                                    for y in range(new_sheet[pair_sheet[1]][0], new_sheet[pair_sheet[1]][1]+1):
                                        if heat_map[x][y] == 0:
                                            anti_parallel.append([x,y])
                                            anti_parallel.append([y,x])

                            for pair_sheet in parallel_sheet_idx:
                            #print(pair_sheet)
                                for x in range(new_sheet[pair_sheet[0]][0], new_sheet[pair_sheet[0]][1]+1):
                                    for y in range(new_sheet[pair_sheet[1]][0], new_sheet[pair_sheet[1]][1]+1):
                                        if heat_map[x][y] == 0:
                                            parallel.append([x,y])
                                            parallel.append([y,x])
                        else:
                          f = open("Only1or0_strand.txt", "a")
                          f.write(pdb_filename+'_' + chain_id + '\n')
                          f.close()

                        for pos in helix_list:
                            mask[pos[0]][pos[1]] = 1
                            #mask[pos[0]][pos[1]] = answer[pos[0]][pos[1]]
                        for pos in sheet_list:
                            mask[pos[0]][pos[1]] = 2
                        for pos in anti_parallel:
                            mask[pos[0]][pos[1]] = 3
                        for pos in parallel:
                            mask[pos[0]][pos[1]] = 4

                        cutted_mask = mask[longest[0]:longest[-1], longest[0]:longest[-1]]
                        cutted_mask = cutted_mask.astype('uint8')
                        image_msk = Image.fromarray(cutted_mask)
                        image_msk.save(gt_dir + '/' + pdb_code + '_' + chain_id +'_gt.png')


                        color = [[255,0,0], [0,255,0], [0,0,255], [255,255,255]]
                        color_mask = np.zeros((cutted_mask.shape[0], cutted_mask.shape[1], 3), dtype='uint8')
                        for i in range(1, 5):
                            color_mask[np.where(cutted_mask==i)] = color[i-1]


                        image_msk = Image.fromarray(color_mask)
                        image_msk.save(color_dir + '/' + pdb_code + '_' + chain_id + '_color.png')
                    
                    else:
                        f = open("Length_problem.txt", "a")
                        f.write(pdb_code + '_' + chain_id +'\n')
                        f.close()
                else:
                    f = open("Big_problem.txt", "a")
                    f.write(pdb_code + '_' + chain_id +'\n')
                    f.close()
            else:
                f = open("NotProteinChain_problem.txt", "a")
                f.write(pdb_code + '_' + chain_id +'\n')
                f.close()

