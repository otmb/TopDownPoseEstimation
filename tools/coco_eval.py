# Refarences
# https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

resFile='vitpose-b256x192_fp16_results.json'
#resFile='vitpose_s256x192_wholebody_fp16_results.json'
#resFile='vitpose_b256x192_wholebody_fp16_results.json'

annType = ['segm','bbox','keypoints']
annType = annType[2]
prefix = 'person_keypoints' if annType=='keypoints' else 'instances'
print('Running demo for *%s* results.'%(annType))

dataDir='.'
dataType='val2017'
annFile = '%s/annotations/%s_%s.json'%(dataDir,prefix,dataType)
cocoGt=COCO(annFile)
cocoDt=cocoGt.loadRes(resFile)
imgIds=sorted(cocoGt.getImgIds())

cocoEval = COCOeval(cocoGt,cocoDt,annType)
cocoEval.params.imgIds = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
