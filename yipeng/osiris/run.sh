#!/bin/bash
#SBATCH --partition=main
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=rtx8000:1
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH -o /network/scratch/y/yipeng.zhang/logs/slurm-%j.out
#SBATCH -e /network/scratch/y/yipeng.zhang/logs/slurm-%j.err

module --quiet purge
module load anaconda/3
module load cuda/11.6
conda activate ucl

##################################################################################

mkdir -p $SLURM_TMPDIR/data
mkdir -p $SLURM_TMPDIR/models
cp -r $SCRATCH/data/cifar-100-python $SLURM_TMPDIR/data

export MASTER_PORT=$(expr 12345 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR="127.0.0.1"
unset CUDA_VISIBLE_DEVICES

##################################################################################

seed="6"
num_tasks="5"
num_epochs="200"
print_freq="200"
buffer_size="500"

cd $HOME/ucl/ucl-mem
out_dir="c${num_tasks}r_od_s${seed}"

torchrun --master_port=$MASTER_PORT \
	--master_addr=$MASTER_ADDR \
	$HOME/ucl/ucl-mem/train.py \
	--dataset cifar100 \
	--num_workers 2 \
	--order task_iid \
	--model continual_simclr_spscdd+_sym \
	--num_tasks "$num_tasks" \
	--epochs "$num_epochs" \
	--batch_size 256 \
	--eval_batch_size 64 \
	--knn_k 200 \
	--data_dir "$SLURM_TMPDIR"/data/cifar-100-python \
	--optimizer sgd \
	--learning_rate_weights 0.03 \
	--learning_rate_biases 0.03 \
	--weight_decay 5e-4 \
	--save_freq 1 \
	--print_freq "$print_freq" \
	--projector 2048-128 \
	--predictor 2048-128 \
	--order_dir "$SLURM_TMPDIR"/data/cifar-100-python/places_order_r0 \
	--save_dir "$SLURM_TMPDIR"/models/"$out_dir" \
	--buffer_size "$buffer_size" \
	--replay_mode concat \
	--seed "$seed" \
	--w_simclr 1 \
	--w_additional 0.5 0.5  \
	--p 0.25 \
	--debug_overfit \
	--distributed

python $HOME/ucl/ucl-mem/knn_eval.py \
	--dataset cifar100 \
	--num_classes 100 \
	--num_tasks "$num_tasks" \
	--data_dir "$SLURM_TMPDIR"/data/cifar-100-python \
	--ckpt_dir "$SLURM_TMPDIR"/models/"$out_dir" \
	--order_dir "$SLURM_TMPDIR"/data/cifar-100-python/places_order_r0 \
	--mode mt_wo_hist \
	--norm gn \
	--act mish \
	--limit_visibility na

cp -r "$SLURM_TMPDIR"/models/"$out_dir" "$SCRATCH"/ucl/models/c5r