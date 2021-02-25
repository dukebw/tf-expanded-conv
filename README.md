# TF Expanded Conv Example


## Setup

```
pip install -r requirements.txt
```


## Usage

```
PYTHONPATH=/path/to/tensorflow-models/research/slim:. python ./tf_expanded_conv.py
tensorflowjs_converter --input_format=tf_saved_model --output_node_names='out' --saved_model_tags=serve ./tf_expanded_conv/ ./tf_expanded_conv/tfjs
```
