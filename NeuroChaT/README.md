# NeuroChaT

NeuroChaT is an open-source neuron characterisation toolbox primarily developed by Md. Nurul Islam under supervision of Shane O'Mara.

## Installation

Python version 3.5 upwards is required to install neurochat. Installation steps are listed in detail below:

### Option 1: Use Pip

Open command prompt and type/paste the following. It is recommended to install neurochat to a virtual environment (E.g. using virtualenv), if doing so, activate it before typing these commands.

```
git clone https://github.com/shanemomara/omaraneurolab.git
cd omaraneurolab
cd NeuroChaT
pip install .
python neurochat_gui\neurochat_ui.py
```

### Option 2: Use Pip, but don't install NeuroChaT

Open command prompt and type/paste the following.

```
git clone https://github.com/shanemomara/omaraneurolab.git
cd omaraneurolab
cd NeuroChaT
pip install -r requirements.txt
python modify_neuro_path.py
python neurochat_gui\neurochat_ui.py
```

This method only allows the GUI program to function, any other file will need to modify the python path to use neurochat.

## Documentation

See the docs folder for a pdf user guide which should be the first port of call. There is also html docs available for the neurochat package, which can be accessed by opening docs/index.html in a browser, or from the NeuroChaT UI help menu.
