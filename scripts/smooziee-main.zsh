#!/usr/bin/zsh

################################################################################
# smooziee-main.zsh
################################################################################

### fuctions
function usage()
{
  cat <<EOF
  "smooziee-main.zsh"

  Options:
    -h           print usage

    --setup      set up for analize
        \$1: "Tananka_May2018_0509_2322" for example
        \$2: new directory

EOF

}

function mk_setup()
{
  # $1 "Tananka_May2018_0509_2322" for example
  # $2 new directory
  mkdir -p $2
  SCAN_DIR=(`ls $1 | grep PbTe_sq | grep ps | sed s/".ps"//g`)
  for DIR in $SCAN_DIR
  do
    mkdir -p $2/$DIR
  done

}

### zparseopts
local -A opthash
zparseopts -D -A opthash -- h -setup

if [[ -n "${opthash[(i)-h]}" ]]; then
  usage
  exit 0
fi


if [[ -n "${opthash[(i)--setup]}" ]]; then
  # $1 "Tananka_May2018_0509_2322" for example
  # $2 new directory
  mk_setup $1 $2
  exit 0
fi

echo "nothing was executed. check the usage  =>  $(basename ${0}) -h"
exit 1
