python run_summac_precomp.py --model vitc --granularity sentence
python train_summac.py --model vitc --granularity sentence --train_batch_size 16 --num_epochs 10 --nli_labels e


# python run_summac_precomp.py --model mnli-base --granularity paragraph
# python run_summac_precomp.py --model mnli-base --granularity 2sents

# python train_summac.py --model mnli-base --granularity paragraph --train_batch_size 8 --nli_labels e
# python train_summac.py --model mnli-base --granularity 2sents --train_batch_size 8 --nli_labels e


# python train_summac.py --model mnli --granularity sentence --train_batch_size 8 --nli_labels n
# python train_summac.py --model mnli --granularity sentence --train_batch_size 8 --nli_labels ec
# python train_summac.py --model mnli --granularity sentence --train_batch_size 8 --nli_labels en
# python train_summac.py --model mnli --granularity sentence --train_batch_size 8 --nli_labels cn
# python train_summac.py --model mnli --granularity sentence --train_batch_size 8 --nli_labels ecn

# python train_summac.py --model vitc --granularity sentence --train_batch_size 8 --nli_labels e
# python train_summac.py --model vitc --granularity sentence --train_batch_size 8 --nli_labels c
# python train_summac.py --model vitc --granularity sentence --train_batch_size 8 --nli_labels n
# python train_summac.py --model vitc --granularity sentence --train_batch_size 8 --nli_labels ec
# python train_summac.py --model vitc --granularity sentence --train_batch_size 8 --nli_labels en
# python train_summac.py --model vitc --granularity sentence --train_batch_size 8 --nli_labels cn
# python train_summac.py --model vitc --granularity sentence --train_batch_size 8 --nli_labels ecn


# python train_summac.py --model anli --granularity sentence --train_batch_size 8 --nli_labels e
# python train_summac.py --model anli --granularity sentence --train_batch_size 8 --nli_labels c
# python train_summac.py --model anli --granularity sentence --train_batch_size 8 --nli_labels n
# python train_summac.py --model anli --granularity sentence --train_batch_size 8 --nli_labels ec
# python train_summac.py --model anli --granularity sentence --train_batch_size 8 --nli_labels en
# python train_summac.py --model anli --granularity sentence --train_batch_size 8 --nli_labels cn
# python train_summac.py --model anli --granularity sentence --train_batch_size 8 --nli_labels ecn

# conda init bash

# conda activate feqa
# python run_baseline.py --model feqa --cut test
# conda activate questeval
# python run_baseline.py --model questeval --cut test
# conda deactivate

# python train_summac.py --model anli --bins even100 --granularity sentence --train_batch_size 8 --nli_labels e
# python train_summac.py --model snli-base --bins even100 --granularity sentence --train_batch_size 8 --nli_labels e
# python train_summac.py --model snli-large --bins even100 --granularity sentence --train_batch_size 8 --nli_labels e
# python train_summac.py --model mnli-base --bins even100 --granularity sentence --train_batch_size 8 --nli_labels e
# python train_summac.py --model vitc-base --bins even100 --granularity sentence --train_batch_size 8 --nli_labels e
# python train_summac.py --model vitc-only --bins even100 --granularity sentence --train_batch_size 8 --nli_labels e

# python run_summac_precomp.py --model decomp --granularity sentence
# python train_summac.py --model decomp --granularity sentence --train_batch_size 8 --nli_labels e

# python run_summac_precomp.py --model anli --granularity sentence
# python run_summac_precomp.py --model snli-large --granularity sentence
# python run_summac_precomp.py --model snli-base --granularity sentence
# python run_summac_precomp.py --model mnli-base --granularity sentence
# python run_summac_precomp.py --model vitc-base --granularity sentence
# python run_summac_precomp.py --model vitc-only --granularity sentence

# python train_summac.py --model vitc --bins even100 --granularity sentence --train_batch_size 8 --nli_labels c
# python train_summac.py --model vitc --bins even100 --granularity sentence --train_batch_size 8 --nli_labels n
# python train_summac.py --model vitc --bins even100 --granularity sentence --train_batch_size 8 --nli_labels ec
# python train_summac.py --model vitc --bins even100 --granularity sentence --train_batch_size 8 --nli_labels en
# python train_summac.py --model vitc --bins even100 --granularity sentence --train_batch_size 8 --nli_labels cn
# python train_summac.py --model vitc --bins even100 --granularity sentence --train_batch_size 8 --nli_labels ecn



# python train_summac.py --model vitc --granularity sentence --train_batch_size 8 --nli_labels e
# python train_summac.py --model vitc --granularity sentence --train_batch_size 8 --nli_labels c
# python train_summac.py --model vitc --granularity sentence --train_batch_size 8 --nli_labels n
# python train_summac.py --model vitc --granularity sentence --train_batch_size 8 --nli_labels ec
# python train_summac.py --model vitc --granularity sentence --train_batch_size 8 --nli_labels en
# python train_summac.py --model vitc --granularity sentence --train_batch_size 8 --nli_labels cn
# python train_summac.py --model vitc --granularity sentence --train_batch_size 8 --nli_labels ecn


# python run_ec_precompute.py --model mnli --granularity 2sents
# python run_ec_precompute.py --model vitc --granularity 2sents

# python train_summac.py --model snli-large --granularity sentence
# python train_summac.py --model snli-large --granularity sentence --bins even100
# python train_summac.py --model snli-large --granularity paragraph
# python train_summac.py --model snli-large --granularity paragraph --bins even100
# python train_summac.py --model mnli-base --granularity sentence
# python train_summac.py --model mnli-base --granularity sentence --bins even100
# python train_summac.py --model mnli-base --granularity paragraph
# python train_summac.py --model mnli-base --granularity paragraph --bins even100
# python train_summac.py --model mnli --granularity sentence
# python train_summac.py --model mnli --granularity sentence --bins even100
# python train_summac.py --model mnli --granularity paragraph
# python train_summac.py --model mnli --granularity paragraph --bins even100
# python train_summac.py --model anli --granularity sentence
# python train_summac.py --model anli --granularity sentence --bins even100
# python train_summac.py --model anli --granularity paragraph
# python train_summac.py --model anli --granularity paragraph --bins even100
# python train_summac.py --model vitc-base --granularity sentence
# python train_summac.py --model vitc-base --granularity sentence --bins even100
# python train_summac.py --model vitc-base --granularity paragraph
# python train_summac.py --model vitc-base --granularity paragraph --bins even100
# python train_summac.py --model vitc --granularity sentence
# python train_summac.py --model vitc --granularity sentence --bins even100
# python train_summac.py --model vitc --granularity paragraph
# python train_summac.py --model vitc --granularity paragraph --bins even100
# python train_summac.py --model vitc-only --granularity sentence
# python train_summac.py --model vitc-only --granularity sentence --bins even100
# python train_summac.py --model vitc-only --granularity paragraph
# python train_summac.py --model vitc-only --granularity paragraph --bins even100
