NOTE: The $ symbol means that the text that follows must be executed in terminal. All other instructions without a $ you need to do manually (e.g. downloading some stuff and putting it into some folder).

01. $ conda create -n rt_gene_standalone python=3.8
02. $ conda activate rt_gene_standalone
03. $ conda install tensorflow-gpu numpy scipy tqdm pillow opencv matplotlib<br>
(depending on what python=3.8 installed, python might get downgraded to 3.8.15 in this step)
04. $ conda install -c pytorch torchvision
05. $ pip uninstall typing_extensions
06. $ pip uninstall fastapi
07. $ pip install --no-cache fastapi<br>
The above 3 steps fix an error that prevents importing ParamSpec from typing_extensions. Step 7 will probably throw some errors regarding dependency issues with tensorflow, but that doesn't matter, because rt_gene allows to use torch instead.
08. $ git clone https://github.com/Frankninetytwo/rt_gene.git
09. $ cd rt_gene
10. $ mkdir OutputImages
11. $ mkdir Frames<br>
**NOTE**: Do not put anything inside the above created folder. First, the contents get deleted multiple times when the feature extraction script is executed. Also, if there is another file inside such folder then that might confuse the program leading to undefined behavior (analyzing files that are not meant to be analyzed, thinking there are more video frames than there really are, etc.).

Feature extraction works exactly the same way as in my L2CS-Net repo. It's explained at the end of that repo's README.md. Note that rt_gene will download some model files when it is executed for the first time.<br>

For the output format (.csv files) also refer to my L2CS-Net repo.
