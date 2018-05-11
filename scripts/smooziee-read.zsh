#!/usr/bin/zsh

################################################################################
# smooziee-file.zsh read various files
################################################################################

### fuctions
function usage()
{
  cat <<EOF
  "smooziee-file.zsh" read from various files

  Options:
    -h           print usage

    --read       read filas
        \$1: filename

EOF

}

function filetype()
{
  ##### $1: filename
  if [ `echo "$1" | grep "gpi"` ] ; then
    echo "gpi"
  fi
}

function read_gpi()
{
  ##### $1: gpi filename
  echo "# qpoints"
  grep "tf_" $1 | grep -v plot | grep "(" | \
    sed -e 's/[^0-9]/ /g' | cut -c 8- | sed -e 's/^[ ]*//g'
}

### zparseopts
local -A opthash
zparseopts -D -A opthash -- h -read

if [[ -n "${opthash[(i)-h]}" ]]; then
  usage
  exit 0
fi


if [[ -n "${opthash[(i)--read]}" ]]; then
  # $1: filename
  FILE_TYPE=`filetype $1`

  if [ "$FILE_TYPE" = "gpi" ]; then
    read_gpi $1
  fi

  exit 0
fi

echo "nothing was executed. check the usage  =>  $(basename ${0}) -h"
exit 1
