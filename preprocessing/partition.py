
import pandas as pd
import argparse
import random

parser = argparse.ArgumentParser(description='Partitioning (random)')
parser.add_argument('-i', action='store', dest='input_file', type=str, required=True, help='Input tree cluster file')
parser.add_argument('-p', action='store', dest='par_size', nargs='+', type=int, help='Partition size, example: 93 93 93 93 93')
parser.add_argument('-o', action='store', dest='output_file', type=str, required=True, help='Output partition index file')
parser.add_argument('-s', action='store', dest='rand_seed', type=str, default=10, help='Random seed (default: 10)')


args = parser.parse_args()

input_file = args.input_file
par_size = args.par_size
output_file = args.output_file
rand_seed = args.rand_seed


# 9 partition
# 77 * 4 + 76 * 5
# 10 partition
# 77 * 1 + 68 * 8 + 67 * 1

# 5 partition
# 
random.seed(rand_seed)

#pd.options.display.max_rows = 999

df = pd.read_csv(input_file, comment='#', names=['Name', 'Cluster', 'Number'], sep=' ')

df_cluster = df.groupby(['Cluster']).agg({'Number':'count'})

df_count = df_cluster.sort_values(['Number'], ascending=False).reset_index()

#print(df.shape)


num_cluster = len(par_size)

# if num_cluster == 9:
#     target_size = [77, 77, 77, 77, 76, 76, 76, 76, 76]
# elif num_cluster == 10:
#     target_size = [77, 68, 68, 68, 68, 68, 68, 68, 68, 67]
# elif num_cluster == 5:
#     #target_size = [137, 137, 138, 138, 138]
#     target_size = [93, 93, 93, 93, 93]

print('# Sample size {}'.format(df.shape[0]))

print('# Number of clusters {}'.format(num_cluster))
print('# Partition size {}'.format(par_size))


cluster_size = df_count.iloc[:num_cluster,:]['Number'].to_list()
partitions = {'Par_{}'.format(i):[df_count.iloc[i,0]] for i in range(num_cluster)}

print('# Initialize clusters')
print(cluster_size)
print(partitions)



for idx, row in df_count.iloc[num_cluster:,:].iterrows():
    cluster = row['Cluster']
    size = row['Number']

    merged = False
    
    while not merged:
        
        par_rand_idx = random.randint(0, num_cluster-1)
        
        if size + cluster_size[par_rand_idx] > par_size[par_rand_idx]:
            continue
        else:
            cluster_size[par_rand_idx] += size
            partitions['Par_{}'.format(par_rand_idx)].append(cluster)
            merged = True

print('\n# Final clusters')
print(cluster_size)
print(partitions)


outfile = open(output_file, 'w')
for i in range(num_cluster):
    par_idx = df[df['Cluster'].isin(partitions['Par_{}'.format(i)])]['Number'].to_list()
    outfile.write(','.join(par_idx)+'\n')
outfile.close()
