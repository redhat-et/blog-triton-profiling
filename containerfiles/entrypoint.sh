#! /bin/bash

set -o errexit  # abort on nonzero exitstatus
set -o nounset  # abort on unbound variable
set -o pipefail # don't hide errors within pipes
# set -o xtrace   # display expanded commands and arguments



if [ -n "$*" ]; then
    exec "$@"
else
    printf "Welcome to the Triton Profiling container\n"
    printf "NVIDIA Nsight Systems\n"
    printf "\tnsys\t\t\tSystems CLI\n"
    printf "\tnsys-ui\t\t\tSystems UI\n"
    printf "NVIDIA Nsight Compute\n"
    printf "\tncu\t\t\tCompute CLI\n"
    printf "\tncu-ui\t\t\tCompute UI\n"
    printf "Jupyter Notebook\n"
    printf "\tstart_jupyter\t\tStart a Jupyter Notebook server\n"
    printf "\t\t\t\tSet NOTEBOOK_PORT to specify the http port used\n"
    printf "\t\t\t\tDefault is %s\n" "${NOTEBOOK_PORT}"
    exec /bin/bash
fi
