
import os
import argparse
import pandas as pd
import numpy as np

def read_blast(filepath):
    

    infile = open(filepath, 'r')
    query_flag = False
    header_flag = False
    write_flag = False

    
    for line in infile:
        
        if line.startswith('Query= '):
            query = line.split()[1]
            data = {}
            data['Query'] = query
            query_flag = True
        
        if line.startswith('Length=') and query_flag:
            query_length = int(line.split('=')[1])
            data['Query_length'] = query_length
            query_flag = False
        
        if line.startswith('***** No hits found *****'):
            data.update({'Subject': np.nan, 'Subject_length': np.nan, 'Max_score':np.nan, 'Total_score':np.nan, 'Evalue': np.nan, 'Alignment_length':np.nan,
                         'N_identity': np.nan, 'P_identity': np.nan, 'N_positive':np.nan, 'P_positive':np.nan, 'N_gaps': np.nan, 'P_gaps': np.nan,
                         'Query_start': np.nan, 'Query_end': np.nan, 'Subject_start': np.nan, 'Subject_end': np.nan, 'Query_seq': np.nan, 'Subject_seq': np.nan})
            yield data



        if header_flag and line.startswith('Length='):
            db_length = int(line.split('=')[1])
            header_flag = False
                
        if header_flag and line.startswith(' '):
            header += '/'+line.strip()[1:]
        
        if header_flag and (not line.startswith(' ')):
            header += line.strip()

        if line.startswith('>'):
            header = line.strip()[1:]
            query_start_last = 1000
            query_end_last = 0
            subject_start_last = 1000
            subject_end_last = 0
            header_flag = True
            
        
        if line.startswith(' Score ='):
            max_score = float(line.split()[2])
            total_score = int(line.split()[4][1:-2])
            evalue = float(line.split()[7][:-1])
        
        if line.startswith(' Identities ='):
            line_split = line.split()
            aln_length = int(line_split[2].split('/')[1])
            nid = int(line_split[2].split('/')[0])
            pid = float(line_split[3][1:-3]) / 100
            npos = int(line_split[6].split('/')[0])
            ppos = float(line_split[7][1:-3]) / 100
            ngap = int(line_split[10].split('/')[0])
            pgap = float(line_split[11][1:-2]) / 100


        if line.startswith('Query  '):
            query_start = int(line.split()[1])
            query_end = int(line.split()[3])
            if query_start < query_start_last:
                query_start_last = query_start
            if query_end > query_end_last:
                query_end_last = query_end

        if line.startswith('Sbjct  '):
            db_start = int(line.split()[1])
            db_end = int(line.split()[3])
            if db_start < subject_start_last:
                subject_start_last = db_start
            if db_end > subject_end_last:
                subject_end_last = db_end
            
            a_sum = (query_end_last - query_start_last + 1) + (subject_end_last - subject_start_last + 1) + ngap
            if a_sum == aln_length*2:
                write_flag = True

        if write_flag:
            data.update({'Subject': header, 'Subject_length': db_length, 'Max_score':max_score, 'Total_score':total_score, 'Evalue': evalue, 'Alignment_length':aln_length,
                            'N_identity': nid, 'P_identity': pid, 'N_positive':npos, 'P_positive':ppos, 'N_gaps': ngap, 'P_gaps': pgap,
                            'Query_start': query_start_last, 'Query_end': query_end_last, 'Subject_start': subject_start_last, 'Subject_end': subject_end_last})
            yield data
            write_flag = False

    
    infile.close()    


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Extract blast results')
    parser.add_argument('-i', action='store', dest='inputfile', nargs='+', required=True, help='Input blast file')
    parser.add_argument('-o', action='store', dest='outputfile', required=True, help='Output csv file')
    #parser.add_argument('-v', action='version', dest='version', version='%(prog)s 1.0')
    parser.add_argument('-e', action='store', dest='threshold_evalue', type=int, default=1e-100, help='Thredshold for evalue')
    args = parser.parse_args()
    inputfile = args.inputfile
    threshold = args.threshold_evalue
    outputfile = args.outputfile

    df = pd.DataFrame()

    for filename in inputfile:

        for data in read_blast(filename):
        
            df = pd.concat([df, pd.DataFrame(data, index=[0])], axis=0, ignore_index=True)
    df.to_csv(outputfile, index=False)

    df_tmp = df.sort_values(['Query_length', 'Evalue'], ascending=[False, True])
    node_list = df_tmp['Query'].unique()

    #clusters = dict()
    #for node in node_list:
        #keep = True
        #if len(clusters) == 0:
            #clusters.append(node)
            #clusters[node] = []
            #print(f'# {node} is the first cluster')
        #else:
            #for knode in clusters.keys():
                #knode = clusters[idx]
                #print(node, knode)
                #evalue = df_tmp[(df_tmp['Query'] == node) & (df_tmp['Subject'] == knode)]['Evalue']
                #if len(evalue) == 0 or evalue.values[0] > threshold:
                    #continue
                #elif evalue.values[0] <= threshold:
                    #keep = False
                    #clusters[knode].append(node)
                    #print(f'# {knode} is added to {node}: {evalue.values[0]}')
                    #break
            #if keep:
                #clusters[node] = []
                #print(f'# {node} is a new cluster')

    #print(len(clusters))
    #print(clusters)
    #for key in clusters.keys():
        #f = open("Dataset1and2_noRedundancy.txt", "a")
        #f.write(key+'\n')
        #f.close()
