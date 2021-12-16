# cs765
CS765 Final Project
------
# run
After cloning this repo, you need to find a dataset. Currently this tool supports labeled high-dimensional data with csv format.
It should have a header, with one "label" column.
For example, download from https://www.kaggle.com/oddrationale/mnist-in-csv and put it in `./static` folder. Then enter the following commands:
```
# on linux
# python >= 3.9
$ python3 -m venv venv
$ . ./venv/bin/activate
$ pip3 install -r req.txt
$ python3 vis-dash/app.py
```

# Used packages
```
Brotli==1.0.9
click==8.0.3
colorama==0.4.4
dash==2.0.0
dash-bootstrap-components==1.0.1
dash-core-components==2.0.0
dash-html-components==2.0.0
dash-table==5.0.0
Flask==2.0.2
Flask-Compress==1.10.1
install==1.3.5
itsdangerous==2.0.1
Jinja2==3.0.3
joblib==1.1.0
llvmlite==0.37.0
MarkupSafe==2.0.1
numba==0.54.1
numpy==1.20.3
pandas==1.3.5
plotly==5.4.0
pynndescent==0.5.5
python-dateutil==2.8.2
pytz==2021.3
scikit-learn==1.0.1
scipy==1.7.3
six==1.16.0
tenacity==8.0.1
threadpoolctl==3.0.0
tqdm==4.62.3
umap-learn==0.5.2
Werkzeug==2.0.2

```