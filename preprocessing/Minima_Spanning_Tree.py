import pandas as pd
import networkx as nx
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

def build_tree(df, threshold, compare):

    nodes = df['Query'].unique()
    if compare == 'le':
        df_edge_t = df[(df['Evalue'] <= threshold) & (df['Query'] != df['Subject'])].reset_index(drop=True)
    elif compare == 'lt':
        df_edge_t = df[(df['Evalue'] < threshold) & (df['Query'] != df['Subject'])].reset_index(drop=True)
    elif compare == 'ge':
        df_edge_t = df[(df['Evalue'] >= threshold) & (df['Query'] != df['Subject'])].reset_index(drop=True)
    elif compare == 'gt':
        df_edge_t = df[(df['Evalue'] > threshold) & (df['Query'] != df['Subject'])].reset_index(drop=True)

    G = nx.Graph()
    G.add_nodes_from(nodes)

    edges = list(zip(df_edge_t['Query'], df_edge_t['Subject']))
    G.add_edges_from(edges)
    print('# Number of clusters:', nx.number_connected_components(G))

    return G


def write_result(G, output_file, blast_file, thres):
    outfile = open(output_file, 'w')
    outfile.write('# Python networkx\n# BLAST file {}\n# Threshold {}\n# Output file\n'.format(blast_file, thres, output_file))

    for idx, component in enumerate(nx.connected_components(G)):
        node_list = sorted(list(component))
        for i in node_list:
            #pass
            outfile.write('ClustID {} {}\n'.format(idx, i))

    outfile.close()


def plot_component2(df_tmp, G_sub, savefig=False, **kwargs):
    #df_tree = pd.DataFram


    #print('# Number of clusters:', nx.number_connected_components(G_sub))
    #print('# Component size:', len(G_sub.nodes()))

    #df_tree = pd.concat([df_tree, write_df(G)], axis=0, ignore_index=True)
    print('start plot')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18,8))
    nx.draw(G_sub, with_labels=True, node_size=3000, node_color='#2437e0', font_weight='bold', font_color='white', font_size=7, ax=ax1)
    ax1.set_title('Number of nodes: {}'.format(len(G_sub.nodes())))

    print('plt heatmap')
    seq_len = df_tmp[['Query', 'Query_length']].drop_duplicates().set_index('Query').to_dict()['Query_length']
    sorted_node = df_tmp.sort_values('Query_length', ascending=False)['Query'].unique()
    df_tmp_pivot = pd.pivot_table(data=df_tmp, index='Query', columns='Subject', values='Evalue', aggfunc='first').fillna(1).reindex(index=sorted_node, columns=sorted_node)

    sns.heatmap(df_tmp_pivot, cmap='crest_r', annot=True, annot_kws={"fontsize":7}, ax=ax2)
    ax2.set_yticklabels([f'{i.get_text()}:{seq_len[i.get_text()]}' for i in ax2.axes.get_yticklabels()])

    # thres_list = [1e-20, 1e-30, 1e-40, 1e-50, 1e-60, 1e-70, 1e-80, 1e-90, 1e-100]
    # size_list = []
    # for thres in thres_list:
    #     rep_list = representative(df_tmp, thres)
    #     size_list.append(len(rep_list))
    # sns.lineplot(x=np.log10(thres_list), y=size_list, ax=ax4, markers='o')
    # ax4.set_xlabel('log10 Threshold')
    # ax4.set_ylabel('Number of representatives')

    
    if savefig:
        print('save fig')
        plt.tight_layout()
        plt.savefig(kwargs['filename'], dpi=1000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build minimum spinnig tree (define_cluster_pdist)', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-b', action='store', dest='blast_file', type=str, required=True, help='Blast list')
    parser.add_argument('-t', action='store', dest='thres', type=float, default=1e-10, help='Threshold for similarity (default = 1e-10)')
    parser.add_argument('-c', action='store', dest='compare', type=str, default='le', choices=['le', 'lt', 'ge', 'gt'], help='Comparison operator for threshold (default = le), \nle: less or equal, \nlt: less than, \nge: greater or equal, \ngt: greater than')
    parser.add_argument('-o', action='store', dest='output_file', type=str, required=True, help='Output cluster file')
    parser.add_argument('-v', action='version', dest='version', version='%(prog)s 1.2')

    args = parser.parse_args()

    blast_file = args.blast_file
    compare = args.compare
    thres = args.thres
    output_file = args.output_file
    
    print('# Threshold: {} {}'.format(compare, thres))

    df = pd.read_csv(blast_file)


    G = build_tree(df, thres, compare)
    write_result(G, output_file, blast_file, thres)

    #G = build_tree(df, thres, compare)
    #Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    #print('b4 loop')
    #for i, G_sub in enumerate(Gcc):
        #df_tmp = df[df['Query'].isin(G_sub) & df['Subject'].isin(G_sub)]
        #G_sub = G.subgraph(G_sub)
        #plot_component2(df_tmp, G_sub, savefig=True, filename='/home/projects/vaccine/people/cheche/thesis/BlastP/tree_png/{0:04}.png'.format(i))
    #nx.draw_networkx(G0, node_size=50, font_size=1, font_color='#530791',edge_color='#4557f7',node_color='#d5adf0')
    #plt.savefig('Biggest_Tree.png', dpi=1000)
    #write_result(G, output_file, blast_file, thres)

