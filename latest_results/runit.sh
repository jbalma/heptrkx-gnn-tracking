#!/bin/bash
source /cray/css/users/jbalma/bin/env_python3.sh
        #csv_path='/lus/scratch/jbalma/temp/heptrkx-gnn-hitgraphs_big_000-np16_16nodes_1ppn_34omp_loadbal_profile/' + csv_name + '.csv'
        #csv_path='/lus/scratch/jbalma/temp/heptrkx-gnn-hitgraphs_big_000-np16_16nodes_1ppn_34omp_shuffle_profile/' + csv_name + '.csv'
        #csv_path='/lus/scratch/jbalma/temp/heptrkx-gnn-hitgraphs_big_000-np16_16nodes_1ppn_68omp_orig/' + csv_name + '.csv'

#PATH_TO_CSV_DIR="/lus/scratch/jbalma/temp/heptrkx-gnn-hitgraphs_big_000-np16_16nodes_1ppn_68omp_orig_profiledio/"

#PATH_TO_CSV_DIR="/lus/scratch/jbalma/temp/heptrkx-gnn-hitgraphs_big_000-np16_16nodes_1ppn_34omp_loadbal_profile/"

#32 workers no load balance
PATH_TO_CSV_ORIG="/lus/scratch/jbalma/temp/heptrkx-gnn-hitgraphs_big_000-np32_32nodes_1ppn_68omp_cray_profiledflopsio_original/"

#32 workers load balancing
PATH_TO_CSV_LB="/lus/scratch/jbalma/temp/heptrkx-gnn-hitgraphs_big_000-np32_32nodes_1ppn_68omp_cray_profiledflopsio_LB/"

python plot_csv.py --csv_path=${PATH_TO_CSV_ORIG} --ranks 32 --output="Original" --max_x=200

python plot_csv.py --csv_path=${PATH_TO_CSV_LB} --ranks 32 --output="Load-Balanced" --max_x=200

