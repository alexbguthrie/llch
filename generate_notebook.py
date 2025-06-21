import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# LLM From Scratch on Google Colab\n",
    "\n",
    "This notebook allows you to train the language model from the `llm_from_scratch` repository using a free Google Colab GPU.\n",
    "\n",
    "**Instructions:**\n",
    "1.  Go to `Runtime` -> `Change runtime type`.\n",
    "2.  Select `T4 GPU` from the `Hardware accelerator` dropdown.\n",
    "3.  Run the cells in order."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Clone the Repository\n",
    "\n",
    "First, we'll clone your GitHub repository to get all the project files."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "%cd /content\n",
    "# Remove the old directory to ensure a fresh clone\n",
    "!rm -rf llch\n",
    "\n",
    "!git clone https://github.com/alexbguthrie/llch.git\n",
    "%cd llch"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Install Dependencies\n",
    "\n",
    "Next, we install all the required Python packages specified in `requirements.txt`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "!pip install -r requirements.txt"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Mount Google Drive (Recommended)\n",
    "\n",
    "Training can take a while, and Colab runtimes can disconnect. To save your progress (your model checkpoints and tokenizers), it's highly recommended to mount your Google Drive. This will allow the script to save files directly to your Drive, where they will persist between sessions.\n",
    "\n",
    "When you run the next cell, you'll be prompted to authorize access to your Google account."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from google.colab import drive\n",
    "drive.mount('/content/drive')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Start Training\n",
    "\n",
    "Now we're ready to train! The command below will start the training process using the `config.yaml` file from the repository.\n",
    "\n",
    "We will override a few key settings for the Colab environment:\n",
    "- `training.device`: We'll set this to `cuda` to use the GPU.\n",
    "- `training.checkpoint_dir`: We'll point this to a folder in your Google Drive to save the model.\n",
    "- `tokenizer.tokenizer_path`: We'll also save the tokenizer to your Google Drive.\n",
    "\n",
    "**Note**: The first time you run this, it will create the `llm_from_scratch_colab` folder in your Google Drive."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "!PYTHONPATH=. python train.py \\\n",
    "  --training.device cuda \\\n",
    "  --training.checkpoint_dir /content/drive/MyDrive/llm_from_scratch_colab/checkpoints \\\n",
    "  --tokenizer.tokenizer_path /content/drive/MyDrive/llm_from_scratch_colab/tokenizer.json"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Generate Text (After Training)\n",
    "\n",
    "Once training is complete (or you've stopped it), you can use your best saved model to generate text.\n",
    "\n",
    "This command uses the `--generation.generate` flag and points to the `best_model.pt` that was saved in your Google Drive."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "!PYTHONPATH=. python train.py \\\n",
    "  --generation.generate \\\n",
    "  --training.resume /content/drive/MyDrive/llm_from_scratch_colab/checkpoints/best_model.pt \\\n",
    "  --generation.prompt \"The secret to happiness is\""
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.10.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open('Run_in_Colab.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print("Generated Run_in_Colab.ipynb") 