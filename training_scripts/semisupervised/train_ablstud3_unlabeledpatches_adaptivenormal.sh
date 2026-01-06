#!/bin/bash
#SBATCH -J crlm-supervised-ablationstudy1
#SBATCH -t 7:00:00
#SBATCH -p gpu_a100
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nicolasjaegergallego@gmail.com


echo "Job id: $SLURM_JOBID"
echo "Nodes: $SLURM_NODELIST"
echo "GPUS assigned:"
nvidia-smi

source $HOME/project_env/bin/activate

# Copy recursively. Created a folder my_dataset
cp -r /scratch-shared/njagergallego/data/lvl3_labeled_normalized "$TMPDIR"/dataset_lvl3
cp -r /scratch-shared/njagergallego/data/lvl4_labeled_normalized "$TMPDIR"/dataset_lvl4
cp -r /scratch-shared/njagergallego/data/lvl5_labeled_normalized "$TMPDIR"/dataset_lvl5
cp -r /scratch-shared/njagergallego/data/lvl4_unlabeled_normalized "$TMPDIR"/lvl4_unlabeled_normalized
mkdir -p "$TMPDIR"/src

# Copy only necessary folders from src
cp -r $HOME/src/config "$TMPDIR"/src/
cp -r $HOME/src/networks "$TMPDIR"/src/
cp -r $HOME/src/preproc_scripts "$TMPDIR"/src/
cp -r $HOME/src/training_scripts "$TMPDIR"/src/
cp -r $HOME/src/utils "$TMPDIR"/src/

# Copy root-level files from src
cp $HOME/src/__init__.py "$TMPDIR"/src/
cp $HOME/src/train_mt.sh "$TMPDIR"/src/
cp $HOME/src/helpers.py "$TMPDIR"/src/

# Create output directory in TMPDIR (not in src)
mkdir -p "$TMPDIR"/outputs

# Go into the folder containing the code
cd $TMPDIR

# Create the final destination directory if it doesn't exist
mkdir -p "$HOME"/results

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Start a background process to copy every minute (60 seconds)
(
  while true; do
    # copy all files and subfolders from outputs/ into results/
    rsync -a "$TMPDIR/outputs/" "$HOME/results/"
    sleep 60
  done
) &
COPY_PID=$!
trap 'kill $COPY_PID 2>/dev/null || true' EXIT
  
python -m src.training_scripts.semisupervised.adaptivePatchFilteringNormal \
    --split 1 \
    --workers 4 \
    --patience 10000 \
    --output_dir "$TMPDIR"/outputs \
    --hpc

# Kill the background copying process once the training is done
kill $COPY_PID

# Final copy to ensure the very last changes are also saved
cp -r "$TMPDIR"/outputs/* "$HOME"/results/