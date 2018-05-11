#!/usr/bin/zsh

################################################################################
# plot various figures
################################################################################

### fuctions
function usage()
{
  cat <<EOF
  "smooziee-plot.zsh" makes various plots

  Options:
    -h           print usage

    --meV_a4_raw make meV_a4 raw data plot
        \$1: "PbTe_sq_GX711_p0p25_RT_4" for example

EOF

}


function mk_meV_a4()
{
  ##### $1: raw data name  ex) "PbTe_sq_GX711_p0p25_RT_4"

}



### zparseopts
local -A opthash
zparseopts -D -A opthash -- h -meV_a4_raw

if [[ -n "${opthash[(i)-h]}" ]]; then
  usage
  exit 0
fi

if [[ -n "${opthash[(i)--meV_a4_raw]}" ]]; then
  mk_band_conf $1
  exit 0
fi

echo "nothing was executed. check the usage  =>  $(basename ${0}) -h"
exit 1
