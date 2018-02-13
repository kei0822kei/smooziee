#!/usr/bin/zsh

################################################################################
# deals with the full data in the df directory
################################################################################

### variables
SCRIPT_DIR=$(cd $(dirname $0); pwd)

### fuctions
function usage()
{
  cat <<EOF
  convenient tools about "qstat"

  Options:
    -h       print usage

    --raw    smooziee-phscat.py --raw
        \$1: df path
        \$2: output direcrtory

    --peak   output executing job infomation
        \$1: df path
        \$2: output direcrtory
        \$3: order


EOF
}


function get_path()
{
  ### $1 df directory path
  ls $1 | grep GXL | grep -v var
}

function concat_fig()
{
  ### $1 all fig arr
  ### $2 savefig

  FIG=(`echo $1`)
  convert +append $FIG[1] $FIG[2] $FIG[3] $FIG[4] hoge1.png
  convert +append $FIG[5] $FIG[6] $FIG[7] $FIG[8] hoge2.png
  convert +append $FIG[9] $FIG[10] $FIG[11] $FIG[12] hoge3.png
  convert -append hoge1.png hoge2.png hoge3.png hoge4.png
  convert -resize 1280x720 -unsharp 2x1.4+0.5+0 -colors 65 -quality 100 -verbose hoge4.png $2
  rm hoge1.png hoge2.png hoge3.png hoge4.png
}


### zparseopts
local -A opthash
zparseopts -D -A opthash -- h -raw -peak -concat

if [[ -n "${opthash[(i)-h]}" ]]; then
  usage
  exit 0
fi

if [[ -n "${opthash[(i)--raw]}" ]]; then
  ### $1 => df path
  ### $2 => save dir
  mkdir $2
  DATA_PATH=(`get_path $1`)
  for DATA in $DATA_PATH:
  do
    smooziee-phscat.py --filename=$1/$DATA --run_mode='raw' --savefig="$2/${DATA}_raw.png"
  done
  exit 0
fi

if [[ -n "${opthash[(i)--peak]}" ]]; then
  ### $1 => df path
  ### $2 => save dir
  ### $3 => order
  mkdir $2
  DATA_PATH=(`get_path $1`)
  for DATA in $DATA_PATH:
  do
    smooziee-phscat.py --filename=$1/$DATA --run_mode='peak' --order=$3 --savefig="$2/${DATA}_peak.png"
  done
  exit 0
fi

if [[ -n "${opthash[(i)--concat]}" ]]; then

  ### $1 => grep $1
  ### $2 => savefig
  FIG_ALL="`ls -v | grep $1`"
  concat_fig "$FIG_ALL" "$2"
  exit 0
fi

echo "nothing was executed. check the usage  =>  $(basename${0}) -h"
exit 1
