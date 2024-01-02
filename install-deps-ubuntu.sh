#!/usr/bin/env bash
if [[ ! -d $PWD/src/pdmproject ]]; then
    echo -e "\033[1;31m[ERROR]\033[0m must run in pdm-project (Group 3) repo root."
    exit -1
fi

# Find default python3 version
python_install=""
deadsnakes_required=false
default_python3_version=$(python3 --version | grep -oE '[[:digit:]]\.[[:digit:]]+')
apt_pkgs_to_install=""

echo -e "\033[1;37m[INFO]\033[0m Default python3 version is $default_python3_version"

if [[ $default_python3_version == '3.10' ]]; then 
    echo -e "\033[1;32m[OK]\033[0m Python 3.10 available no action necessary"
    python_install="python3"
else
    # Check if python3.10 installed
    echo -e "\033[1;33m[WARN]\033[0m Python 3.10 is required, checking alternative python versions..."
    
    python310_installed=$(dpkg-query --list python3.10 | tail -n +6 | awk -F ' ' '{print $1}')

    if [[ $? -eq 1 ]]; then 
        echo -e "\033[1;33m[WARN]\033[0m Python 3.10 is not availble in your repositoryies. deadsnakes/ppa required."
        deadsnakes_required=true
    else
        echo -e "\033[1;32m[OK]\033[0m Python 3.10 is found in your repositories"
        if [[ $$python310_installed == "ii" ]]; then
            echo -e "\033[1;32m[OK]\033[0m python3.10 already installed no actions neccessary"
            python_install='python3.10'
        else
            echo -e "\033[1;33m[WARN]\033[0m python3.10 is not installed yet"
            read -p "Do you want to install it from your already setup repos? [Yn]" answer

            if [[ $(echo ${answer:0:1} | grep --ignore-case --count "n") -eq 0 ]]; then
                sudo apt install python3.10
                python_install='python3.10'
            else
                echo -e "\033[1;31m[ERROR]\033[0m Python 3.10 is required, stopping setup."
                exit -1
            fi
        fi
    fi
fi
# dpkg --list python3 | tail  -n 1 | awk -F ' ' '{print $3}'

if $deadsnakes_required; then
    echo "Python 3.10 could not be installed via your normal repositories."
    read -p "Do you want to add the deadsnakes/ppa and install python 3.10 to continue? [yN]" answer
    if [[ $(echo ${answer:0:1} | grep --ignore-case --count "y") -eq 1 ]]; then
        sudo add-apt-repository ppa:deadsnakes/ppa
        echo "Refreshing repositories"
        sudo apt update
        sudo apt install python3.10 -y
        python_install='python3.10'
    else
        echo -e "\033[1;31m[ERROR]\033[0m Python 3.10 is required, stopping setup."
        exit -1
    fi
fi


echo -e "\033[1;33m[ACTION]\033[0m Attempting to install $python_install-pip $python_install-venv $python_install-tk"
read -p "Do you want to install? [Yn]" answer
if [[ $(echo ${answer:0:1} | grep --ignore-case --count "n") -eq 0 ]]; then 
    sudo apt install "$python_install-pip" "$python_install-venv" "$python_install-tk" -y
fi

read -p "Do you want to setup a venv and install the requirements? [Yn]" answer
if [[ $(echo ${answer:0:1} | grep --ignore-case --count "n") -eq 0 ]]; then 
    read -rp "What venv name do you want? [venv]" venv_name
    if [[ -z $venv_name ]]; then
        echo "Using default venv name 'venv'"
        venv_name=venv
    fi

    $python_install -m venv $venv_name
    ./$venv_name/bin/pip install pip --upgrade
    ./$venv_name/bin/pip install -r requirements.txt
    ./$venv_name/bin/pip install -e .
fi

echo -e "\033[1;32m[OK]\033[0m Done"