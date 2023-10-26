# Installation

Installation can be done via Anaconda or Miniconda.The steps for installation of Anaconda or Miniconda and of the application libraries are given below.

## Anaconda/Miniconda installation

Miniconda can be installed by downloading the installer for your operating system from the official website: https://docs.conda.io/projects/miniconda/en/latest/index.html. For those who prefer a GUI, Anaconda would be preferable; the installation file is available via the link: https://www.anaconda.com/download

It is recommended that you install mamba to increase installation speed.

```bash
conda install mamba -n base -c conda-forge
```

## Package installations

The steps for installation of the packages are as follows:
In the bash command line, go to the project root folder.

```bash
$ cd ~/SourceCode/src/main
```

Run the Conda command line tool to create a virtual environment and install the packages listed in the `environment.yml` file.

```bash
$ conda env create -f environment.yml
```

After the installation is completed, the backend API server can be started by running the Flask command.

```bash
$ flask run
```

If the API is running locally, the app can be opened via a browser at the URL http://localhost:5000/

# View Tensorboard logs

```bash
$ tensorboard --logdir="./tensorboard_logs"
```
