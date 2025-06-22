import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# LLM From Scratch on Google Colab\n",
    "\n",
    "This notebook trains the language model from the `llm_from_scratch` repository using a free Google Colab GPU.\n",
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
    "## 1. Setup Environment and Get Code\n",
    "\n",
    "This cell clones the GitHub repository. If it's already been cloned, it will pull the latest changes to make sure you have the most recent code."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "%cd /content\n",
    "\n",
    "repo_path = '/content/llch'\n",
    "\n",
    "if os.path.exists(repo_path):\n",
    "    %cd {repo_path}\n",
    "    print('Repository already exists. Pulling latest changes...')\n",
    "    !git pull\n",
    "else:\n",
    "    print('Cloning repository...')\n",
    "    !git clone https://github.com/alexbguthrie/llch.git\n",
    "    %cd {repo_path}\n",
    "\n",
    "print('\\nDone. Current directory:')\n",
    "%pwd"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Install Dependencies\n",
    "\n",
    "Next, we install the Python packages needed to run the code. We only install what's necessary to avoid conflicts with Colab's built-in packages."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# We install specific versions of libraries known to be compatible to avoid errors.\n",
    "!pip install datasets==2.14.5 fsspec==2023.6.0 huggingface-hub==0.22.2 PyYAML"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Mount Google Drive\n",
    "\n",
    "This step connects to your Google Drive so we can save the trained model and tokenizer. You will be asked to authorize this connection."
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
   "source": [
    "## 4. Clean Up Old Files (Important!)\n",
    "\n",
    "Since we fixed the code, the old model and tokenizer files are broken. This cell deletes them from your Google Drive to make sure we train a new, working model."
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "source": [
    "import shutil\n",
    "import os\n",
    "\n",
    "# The path to your project folder on Google Drive\n",
    "colab_dir = '/content/drive/MyDrive/llm_from_scratch_colab'\n",
    "\n",
    "# Define paths for the new, larger model's checkpoints and tokenizer\n",
    "checkpoint_dir = os.path.join(colab_dir, 'checkpoints/gpt_medium')\n",
    "tokenizer_path = os.path.join(colab_dir, 'tokenizers/bpe_gpt2_small.json')\n",
    "\n",
    "# Remove the old checkpoint directory if it exists\n",
    "if os.path.exists(checkpoint_dir):\n",
    "    print(f\"Removing old checkpoint directory: {checkpoint_dir}\")\n",
    "    shutil.rmtree(checkpoint_dir)\n",
    "\n",
    "# Remove the old tokenizer file if it exists\n",
    "if os.path.exists(tokenizer_path):\n",
    "    print(f\"Removing old tokenizer: {tokenizer_path}\")\n",
    "    os.remove(tokenizer_path)\n",
    "\n",
    "print(\"\\nCleanup complete. Ready to train!\")"
   ],
   "metadata": {},
   "execution_count": None,
   "outputs": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Start Training\n",
    "\n",
    "Now we're ready to train! This command will:\n",
    "1. Create a new, correct tokenizer and save it to your Drive.\n",
    "2. Train the model using the Colab GPU.\n",
    "3. Save model checkpoints to your Drive."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# We need to set the PYTHONPATH to include our project directory\n",
    "# We override the config to use Colab-specific paths and run for more epochs.\n",
    "!PYTHONPATH=/content/llch python3 train.py \\\\",
    "  --training.device cuda \\\\",
    "  --training.max_epochs 10 \\\\",
    "  --training.checkpoint_dir /content/drive/MyDrive/llm_from_scratch_colab/checkpoints/gpt_medium \\\\",
    "  --tokenizer.tokenizer_path /content/drive/MyDrive/llm_from_scratch_colab/tokenizers/bpe_gpt2_small.json"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Generate Text (After Training)\n",
    "\n",
    "Once training is complete, you can use your best model to generate text. This command points to the `best_model.pt` that was saved in your Google Drive."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "!PYTHONPATH=/content/llch python3 train.py \\\\",
    "  --generation.generate \\\\",
    "  --training.resume /content/drive/MyDrive/llm_from_scratch_colab/checkpoints/gpt_medium/best_model.pt \\\\",
    "  --generation.prompt \\"The secret to happiness is\\""
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