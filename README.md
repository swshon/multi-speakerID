# multi-speakerID baseline
This is python implementation of multi-target speaker recognition based on i-vector feature. This is also baseline system of the first Multi-target speaker detection and identification Challenge Evaluation (MCE 2018, http://www.mce2018.org )


# Dataset
You can download i-vector dataset if you register and confirmed by MCE 2018 organizer. After download, extract to data folder

# System flow
<img align="center" width="600" src="https://github.com/swshon/multi-speakerID/blob/master/img/multitarget_baseline_v2.png ">

# Performance
If you run the code like

    python mce2018_baseline_dev.py
    
you will see the performance on top-S and top-1 detector as below :

    Dev set score using train set :
    Top S detector EER is 2.01%
    Top 1 detector EER is 13.40% (Total confusion error is 492)

And the code also generate example submission file with name "teamname_fixed_primary.csv" and the format are [test utterance ID],[score],[Closest blacklist speaker ID] per each files. For example

    pnah_431154,0.36613864,33762391
    qtxw_470243,0.39585015,60587769
    ....

# Question
Please email to mce organizer if you have question.
mce@lists.csail.mit.edu or swshon@mit.edu

