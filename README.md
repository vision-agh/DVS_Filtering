Na początku planuje zrealizować model CNN, GCN, ViT (potencjalnie SNN) do klasyfikacji. Struktura będzie następująca:

1. W data/ znajdować się będą moduły do datasetów, preprocessing, augmentacja itd. Pracować będę na danych sparse (przed końcową reprezentacją).
2. Przetworzone dane będę przesyłać do generatora reprezentacji (CNN - event-frame, ViT - event-voxel (czasowo przestrzenny), GCN - event-graph, SNN - coś tam). Każdy generator będzie przypisany do danego modelu.
3. Modele będą zaimplementowane w plikach models. Każdy model będzie miał swoje własne parametry, które będą przekazywane z pliku config.yaml.


W ramach CNN, planuje wykorzystać 3 wersje ResNet, potencjalnie MobileNet aby był mniejszy.

W ramach ViT, planuje wykorzystać MaxVit w 3 rozmiarach 32/48/64 zgodnie z pracą RVT-Event.

W ramach GCN, planuje wykorzystać SplineConv w 3 rozmiarach.

W ramach SNN, model będzie prawdopodbnie taki sam jak CNN, tylko odpowiednie aktywacje.

Taki sam optymalizator do każdego, takie same hiperparametry, takie same metryki. Co do augmentacji, zastosuje również takie same do każdej reprezentacji (więc jakis crop, flip, shift, rotate), który można do każdej z tych reprezentacji zrobić.


Bazuje na RVT-Event oraz implementacji w timm

https://github.com/uzh-rpg/RVT

https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/maxxvit.py#L680



Na razie biblioteki to pytorch i standardowe. Dodaje PyTorch Geometric i potencjalne TorchSNN/SpikingJelly.


Dodatkowo omegaconf


conda create -y -n dvs_fil python=3.9
conda activate dvs_fil

conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install h5py
conda install blosc-hdf5-plugin lightning -c conda-forge

pip install matplotlib tqdm numba scikit-learn wandb pyyaml opencv-python pybind11 omegaconf
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
pip install lightning