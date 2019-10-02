#!/bin/bash

BERT_BASE_DIR="/remote/bones/user/vbalacha/pretrained_bert/uncased_L-24_H-1024_A-24"
NQ_BASELINE_DIR="/remote/bones/user/vbalacha/bert-joint-baseline"
#LOAD_MODEL="output/squad-0.1_bertbase_qrystartend_finetune_lr3e-5/best_model"
#SQUAD_DIR="../datasets/zero-shot-relation-extraction/relation_splits/split_both_1/"
APR_DIR="/remote/bones/user/vbalacha/google-research/fat/fat/fat_bert_nq/files/"
SEQ_LEN=512
INC_UNK=0.02
OUTPUT="/remote/bones/user/vbalacha/google-research/fat/fat/fat_bert_nq/generated_files/sharded_kb_data_text_fact_seperate_features_mc48_mseq${SEQ_LEN}_unk${INC_UNK}"

#OUTPUT="/remote/bones/user/vbalacha/google-research/fat/fat/fat_bert_nq/generated_files/tmpdir"

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
          --create_sep_text_fact_inputs=True \
          --input_data_dir=/remote/bones/user/vbalacha/datasets/ent_linked_nq_new/ \
          --output_data_dir=$OUTPUT \
          --apr_files_dir=$APR_DIR \
          --full_wiki=True \
          --create_sep_text_fact_inputs=True \
          --vocab_file=$NQ_BASELINE_DIR/vocab-nq.txt \
          --do_lower_case=True \
          --merge_eval=False \
          --max_seq_length=$SEQ_LEN \
          --include_unknowns=$INC_UNK > log/textfact_dev_$i$j.log 2>&1 &
    done
    wait
    echo "All Done for this iteration"
done
