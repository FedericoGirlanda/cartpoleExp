# quanser_linear_inverted_pendulum

Quanser Linear Inverted Pendulum Setup


- courseware_resources_LIP : contains documentation for navigating through the software and implement balance and swing up control for the LIP

# Pure Python Control
The software for that is in sw/python

The goal for now is to access the pendulum using the established API for the other 
underactuated lab systems.

### Getting the Requirements from Quanser

The following works on Ubuntu 20.04:

First, get the software source: 

    wget --no-cache https://repo.quanser.com/debian/release/config/configure_repo.sh

    chmod u+x configure_repo.sh

    ./configure_repo.sh

    rm -f ./configure_repo.sh

    sudo apt update


Once the repository is set up, you can use the following command to install the python3 packages:

    sudo apt-get install python3-quanser-apis

If you want the QUARC runtime, execute the command (not required):

    sudo apt-get install quarc-runtime

To see a list of most of the Quanser packages and their status (installed or not) do:

    apt search quanser

Documentation on Quanser API is available here: https://docs.quanser.com/quarc/documentation/python/hardware/index.html

With these steps, you should be able to run the simple test programs in sw/python/.  Further python requirements are listed in requirements.txt

