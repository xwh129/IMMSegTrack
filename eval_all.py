import motmetrics as mm
import numpy as np 
import os
from utils_ds.io import read_results

i=1
accs=[]
names=[]
while i <=49:
    if (os.path.exists("runs/detect/track"+str(i)+"/out_v8.txt")):
        gt_file="datasets/dancetrack/"+str(i)+"/gt/gt.txt"
        ts_file="runs/detect/track"+str(i)+"/out_v8.txt"
        gt=mm.io.loadtxt(gt_file, fmt="mot15-2D", min_confidence=1)
        ts=mm.io.loadtxt(ts_file, fmt="mot15-2D")
        name=os.path.splitext(os.path.basename(ts_file))[0]

        acc=mm.utils.compare_to_groundtruth(gt, ts, 'iou', distth=0.5)

        accs.append(acc)
        names.append(str(i))
    i+=1

metrics = list(mm.metrics.motchallenge_metrics)
mh = mm.metrics.create()
summary = mh.compute_many(accs, metrics=metrics, names=names,generate_overall=True)
result=mm.io.render_summary(summary, formatters=mh.formatters,namemap=mm.io.motchallenge_metric_names)
print("---------------------------------------------------------------------")
print("dataset",result)
