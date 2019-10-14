#!/bin/sh
# Author: David Bielik

desired_folder_nr=-1
folder_prefix="ex0"

print_usage() {
    echo "Zips the required files from each folder into a zip file ready for submission."
    echo " "
    echo "options:"
    echo "-n                specify the folder number to be zipped (e.g. -n 1). Range: [1, 4]"
}

zip_folder() {
    d=$1
    target_file_prefix=${d%/}
    target_py_files="$d${target_file_prefix}_"*".py"
    archive_name="${target_file_prefix}_davidbielik_deborabeuret"
    lab_report="${d}${target_file_prefix}_labreport.pdf"

    # check if the lab report pdf exists
    if [ ! -f $lab_report ]; then
        # if not, continue the loop
        echo "warning: $lab_report not found. Skipping ${target_file_prefix}..."
        return
    fi

    # check if at least 1 target file is present
    for f in $target_py_files ; do
        if [ -e $f ]; then
            break
        else
            echo "warning: ${d}_*.py not found. Skipping ${target_file_prefix}..."
            return
        fi
    done
    echo "Zipping ${target_file_prefix}..."
    # zip the archives
    zip -q -j $archive_name $target_py_files $lab_report
}

# Input
while getopts 'n:' flag; do
  case "${flag}" in
    n) desired_folder_nr=${OPTARG} ;;
    *) print_usage
       exit 1 ;;
  esac
done

# Input
if ! [ $desired_folder_nr -eq -1 ] ; then
    # folder specified, zip specific
    if ! [[ $desired_folder_nr =~ ^[1-4]$ ]] ; then
        echo "error: Not a valid number of exercise. Please choose in the range of [1, 4]." >&2; exit 1
    fi
    target="${folder_prefix}${desired_folder_nr}/"
    zip_folder $target; exit 0;
else
    # folder not specified, zip every folder
    for d in */ ; do
        # wait for exercise folders
        if ! [[ $d =~ ^($folder_prefix)[1-4] ]]; then
            continue
        fi

        zip_folder $d
    done
fi
