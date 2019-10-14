#!/bin/bash

BERT_BASE_DIR="/remote/bones/user/vbalacha/pretrained_bert/wwm_uncased_L-24_H-1024_A-24"
NQ_BASELINE_DIR="/remote/bones/user/vbalacha/bert-joint-baseline"
APR_DIR="/remote/bones/user/vbalacha/google-research/fat/fat/fat_bert_nq/files/"
SEQ_LEN=512
INC_UNK=0.02
ALPHA=0.75
OUTPUT="/remote/bones/user/vbalacha/google-research/fat/fat/fat_bert_nq/generated_files/sharded_kb_data_alpha${ALPHA}_mc48_mseq${SEQ_LEN}_unk${INC_UNK}"


#mkdir -p $OUTPUT
#mkdir -p $OUTPUT/train
#mkdir -p $OUTPUT/dev
#mkdir -p $OUTPUT/pretrain
#mkdir -p $OUTPUT/pretrain/train

#for j in {0..6}
for i in {0..4}
do
    #echo $j
    #for i in {0..49}
    for j in {0..16}
    do
        echo  $i
        echo $j
        nohup python3 -m fat.fat_bert_nq.prepare_nq_data \
          --is_training=False \
          --verbose_logging=False \
          --split=dev \
          --task_id=$i \
          --shard_split_id=$j \
          --input_data_dir=/remote/bones/user/vbalacha/datasets/ent_linked_nq_new/ \
          --output_data_dir=$OUTPUT \
          --pretrain_data_dir=$OUTPUT/pretrain \
          --create_pretrain_data=False \
          --use_random_fact_generator=False \
          --apr_files_dir=$APR_DIR \
          --full_wiki=True \
          --vocab_file=$NQ_BASELINE_DIR/vocab-nq.txt \
          --do_lower_case=True \
          --use_entity_markers=False \
          --merge_eval=False \
          --alpha=${ALPHA} \
          --max_seq_length=$SEQ_LEN \
          --include_unknowns=$INC_UNK > log/alphadev_$i$j.log 2>&1 &
    done
    wait
    echo "All Done for this iteration"
done
