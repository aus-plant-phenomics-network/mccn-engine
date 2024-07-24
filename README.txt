# README 
## _Installing the MCCN and using the demo notebook._

Here are some of the main steps to install dependencies so you are able to use the MCCN demo notebook.


### 1. Install pyenv and pyenv-virtualenv plugin

- To install this pyenv follow this [link](https://github.com/pyenv/pyenv?tab=readme-ov-file#getting-pyenv)
- To install pyenv-virtualenv follow this [link](https://github.com/pyenv/pyenv-virtualenv?tab=readme-ov-file#activate-virtualenv)

### 2. Create a virtual environment using Python 3.12.2
##### Using pyenv virtualenv with pyenv

- Follow this [link](https://github.com/pyenv/pyenv-virtualenv) use pyenv virtualenv with pyenv 

### 3. Activate virtualenv
``
pyenv activate mccn
``

- For more info follow this [link](https://github.com/pyenv/pyenv-virtualenv?tab=readme-ov-file#activate-virtualenv)

### 4. Clone the mccn repository
``git clone https://github.com/aus-plant-phenomics-network/mccn-engine``

### 5. Install poetry following the instructions in the link 

- Follow this [link](https://github.com/python-poetry/install.python-poetry.org)

Note: You might want to add Poetry to the $PATH, copy the $PATH showed when you install Poetry. 

**On MAC**
``
vim ~/.zshrc
``
Paste the $PATH in the first row of the file and save file

### 6. Install dependencies 


``cd mccn-engine``
``poetry install``

### 7. Start the Notebook server
``python -m notebook``



