#!/bin/bash


########################## MAIN SCRIPT ##########################
base_name="${SLURM_SCRIPT%.*}"
config_file="./scripts/cluster/slurm/utilities/slurm_config_${CLUSTER_NAME}.sh"
temp_file="${base_name}_temp.sh"

sed -i "/^#!\/bin\/bash/r ${config_file}" $SLURM_SCRIPT
envsubst < $SLURM_SCRIPT > $temp_file

job_id=$(sbatch $temp_file | awk '{print $4}')
echo "Submitted job with ID: $job_id"
########################## MAIN SCRIPT ##########################

cleanup() {
    echo "Script interrupted. Cleaning up..."
    scancel "$job_id" 2>/dev/null
    echo "Job $job_id has been canceled."
    rm $temp_file
    echo "Temporary file $temp_file has been deleted."
    exit 1
}
trap cleanup SIGINT

# Store the job submission time
start_time=$(date +%s)
# Wait until the job is running
while true; do
    job_status=$(squeue -j "$job_id" -h -o "%T")
    if [ "$job_status" == "RUNNING" ]; then
        echo "Job $job_id is now running."
        end_time=$(date +%s)  # Get the current time when the job starts running
        waiting_time=$((end_time - start_time))  # Calculate the waiting time
        echo "The job waited for $waiting_time seconds before starting."
        sleep 5
        break
    elif [ -z "$job_status" ]; then
        echo "Job $job_id has finished or failed before reaching running state."
        exit 1
    else
        echo "Job $job_id is still in $job_status state. Checking again in 10 seconds..."
        sleep 10
    fi
done

# Plot the real-time output
output_file=$(scontrol show job "$job_id" | awk -F= '/StdOut/ {print $2}' | sed "s/%A/${job_id}/g" | sed "s/%a/1/g")
echo "Tailing output file: $output_file"
tail -f "$output_file"
