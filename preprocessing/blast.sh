#!/bin/bash

HM="/home/projects/vaccine/people/cheche/thesis/BlastP"

INPUT_folder="$HM/smile3_fsa"
DB_folder="$HM/smile3_db"
OUTPUT_folder="$HM/smile3_res"

module load ncbi-blast/2.12.0+

#makeblastdb -in Dataset1.fsa -dbtype prot -out database_dir/database_name

for file in "$INPUT_folder"/*;do
    f=$(echo "${file##*/}");
    filename=$(echo $f| cut  -d'.' -f 1);
    blastp -evalue 1e-10 -query $file -db $DB_folder/smile3 -out $OUTPUT_folder/${filename}_res
done
