export PYTHONPATH=../models/insightface/deploy/
python3 generate_insightface_embedding.py --image-size 112,112 --model ../models/insightface/models/model-r100-ii/model,0 --gpu 0
python3 generate_insightface_embedding.py --image-size 112,112 --model ../models/insightface/models/model-r34-amf/model,0 --gpu 0
python3 generate_insightface_embedding.py --image-size 112,112 --model ../models/insightface/models/model-r50-am-lfw/model,0 --gpu 0
python3 generate_insightface_embedding.py --image-size 112,112 --model ../models/insightface/models/model-y1-test2/model,0 --gpu 0