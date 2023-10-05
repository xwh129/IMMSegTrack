import motmetrics as mm
import numpy as np 
import os
from utils_ds.io import read_results
# Metrics
Width,Height=1920,1080
metrics = list(mm.metrics.motchallenge_metrics)

i = 1

while i<=49:
    filename="runs/segment/track"+str(i)+"/labels"
    gt_file_path="datasets/SNMOT/"+str(i)+"/gt/gt.txt"
    with open(os.path.join(*filename.split("/")[:-1])+"/out_v8.txt","w+") as o_f:
        for dirs,_,files in os.walk(filename):
            files=sorted(files)
            for file in files:
                txt_path=os.path.join(dirs,file)
                frame=file[:-4].split("_")[-1]
                frame=str(int(frame))
                # print(frame)
                with open(txt_path) as f:
                    lines=f.readlines()
                    for line in lines:
                        line_split=line.split()
                        if int(line_split[0])!=0:
                            continue

                        w=int(float(line_split[3])*Width)
                        h=int(float(line_split[4])*Height)

                        x=int(float(line_split[1])*Width)-w/2
                        y=int(float(line_split[2])*Height)-h/2

                        new_line=frame+","+line_split[-1]+","+str(x)+","+str(y)+","+str(w)+","+str(h)+","+str(1)+","+str(1)+","+str(1)
                        o_f.writelines(new_line+"\n")
    #导入gt和ts文件
    gt_file= gt_file_path
    ts_file=os.path.join(*filename.split("/")[:-1])+"/out_v8.txt"
    gt=mm.io.loadtxt(gt_file, fmt="mot15-2D", min_confidence=1)
    ts=mm.io.loadtxt(ts_file, fmt="mot15-2D")
    name=os.path.splitext(os.path.basename(ts_file))[0]
    acc=mm.utils.compare_to_groundtruth(gt, ts, 'iou', distth=0.5)
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=metrics, name=name)
    result=mm.io.render_summary(summary, formatters=mh.formatters,namemap=mm.io.motchallenge_metric_names)
    score=result.split()[32][:-1]
    print(result,float(score))
