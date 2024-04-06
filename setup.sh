#!/bin/bash
set -eu  # Exit on error

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Define relative paths based on the script directory
storage_dir="$script_dir/data/"
extract_dir="$script_dir/data/"

function Svarah_Dataset_Download() {
	if ! test -e $storage_dir/svarah; then
		echo "Download LibriSpeech/test-clean into $storage_dir"
		# If downloading stalls for more than 20s, relaunch from previous state.
		wget -c --tries=0 --read-timeout=20 https://indic-asr-public.objectstore.e2enetworks.net/svarah.tar -P $storage_dir
		tar -xvf $storage_dir/svarah.tar -C $extract_dir
		rm -rf $storage_dir/svarah.tar
	fi
}

Svarah_Dataset_Download
