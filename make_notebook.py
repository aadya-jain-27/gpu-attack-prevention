"""Turns gpu_ids.py (cells marked with '# %%') into a Colab notebook."""
import json, re
text = open("gpu_ids.py").read()
cells = []
for chunk in re.split(r"^# %%", text, flags=re.M)[1:]:
    if chunk.startswith(" [markdown]"):
        body = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in chunk.split("\n")[1:]).strip()
        cells.append({"cell_type": "markdown", "metadata": {}, "source": body})
    else:
        body = chunk.strip("\n").replace("# !pip", "!pip")
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": body})
nb = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4"},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"}}}
json.dump(nb, open("gpu_attack_prevention.ipynb", "w"), indent=1)
print(len(cells), "cells")
