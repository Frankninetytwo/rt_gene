01. $ conda create -n rt_gene_standalone python=3.8
02. $ conda activate rt_gene_standalone
03. $ conda install tensorflow-gpu numpy scipy tqdm pillow opencv matplotlib
(depending on what python=3.8 installed, python might get downgraded to 3.8.15 in this step)
04. $ conda install -c pytorch torchvision
05. $ pip uninstall typing_extensions
06. $ pip uninstall fastapi
07. $ pip install --no-cache fastapi
(the above 3 steps fix an error that prevents importing ParamSpec from typing_extensions)
08. git clone https://github.com/Frankninetytwo/rt_gene.git
09. $ cd rt_gene
10. $ mkdir InputImages
