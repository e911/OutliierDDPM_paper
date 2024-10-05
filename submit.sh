#!/bin/bash

job_file="job.sh"

dir="out"
mkdir -p $dir

job_name="Oct-5"
export job_name

out_file=$dir/out_$job_name.out
error_file=$dir/err_$job_name.err

echo $job_name
sbatch -J $job_name -o $out_file -e $error_file $job_file