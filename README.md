## For fisheye distortion correction only

```sh
cd fisheye
python unfish.py <input_video>
```

## For fisheye distortion correction only

```sh
cd fisheye
python unfish.py <input_video>
```

## For demo web app

> object detection/tracking + fisheye undistortion included

#### Local environment

<details><summary>Note</summary>

- For non-GPU users, please install CPU version of PyTorch first

```
pip install -i https://download.pytorch.org/whl/cpu torch torchvision
```

</details>

```
pip install -r requirements.txt
```

| Streamlit        |
|-------------------------| 
| `streamlit run webapp.py` |

