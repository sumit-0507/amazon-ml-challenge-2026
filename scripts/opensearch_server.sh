#!/usr/bin/env bash
# OpenSearch server for the entity-resolution indices, run as a Slurm job.
#
#   mkdir -p logs && sbatch scripts/opensearch_server.sh     (from the repository root)
#
# Listens on 127.0.0.1:${OS_PORT} of the compute node only (security plugin is
# off, so it is not exposed to other cluster users). Run clients inside the job:
#
#   srun --jobid=<job id> --overlap python scripts/index_opensearch.py --split train
#
# Index data lives in ${OS_DATA} on /scratch and survives the job; resubmitting
# this script brings the same indices back up.
#
#SBATCH --job-name=opensearch
# The GPU is unused by OpenSearch, but the partition requires one and grants
# CPUs per slice (2g.48gb -> 12 CPUs).
#SBATCH --partition=gpu_medium
#SBATCH --qos=gpu_medium
#SBATCH --gres=gpu:2g.48gb:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/opensearch-slurm-%j.out

set -euo pipefail

OS_HOME=${OS_HOME:-/scratch/$USER/opensearch/opensearch-3.2.0}
OS_DATA=${OS_DATA:-/scratch/$USER/opensearch/data}
OS_LOGS=${OS_LOGS:-/scratch/$USER/opensearch/logs}
OS_PORT=${OS_PORT:-9277}
OS_HEAP=${OS_HEAP:-16g}  # rest of the job's memory is left to the OS page cache

mkdir -p "$OS_DATA" "$OS_LOGS"
echo "opensearch on $(hostname):$OS_PORT (localhost only), data=$OS_DATA"

export OPENSEARCH_JAVA_OPTS="-Xms$OS_HEAP -Xmx$OS_HEAP"
# native Faiss/nmslib libraries of the k-NN plugin (what opensearch-tar-install.sh does)
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$OS_HOME/plugins/opensearch-knn/lib"
exec "$OS_HOME/bin/opensearch" \
    -Ecluster.name=amz-er \
    -Enode.name=amz-er-1 \
    -Epath.data="$OS_DATA" \
    -Epath.logs="$OS_LOGS" \
    -Enetwork.host=127.0.0.1 \
    -Ehttp.port="$OS_PORT" \
    -Etransport.port=9377 \
    -Ediscovery.type=single-node \
    -Eplugins.security.disabled=true
