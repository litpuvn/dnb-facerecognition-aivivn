export PYTHONPATH=../insightface/deploy/
python3 insightface_embedding_distance.py --model ../insightface/models/model-r100-ii/model,0
python3 insightface_embedding_distance.py --model ../insightface/models/model-r34-amf/model,0
python3 insightface_embedding_distance.py --model ../insightface/models/model-r50-am-lfw/model,0
python3 insightface_embedding_distance.py --model ../insightface/models/model-y1-test2/model,0