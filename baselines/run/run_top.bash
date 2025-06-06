for size in 176 500 528 1500; do
    for mode in GrdStation ISL; do
        nohup python top.py --mode $mode --size $size
        # nohup python top.py --intensity $intensity --mode $mode --top-percentage 0.1
    done
done

nohup python top.py --mode GrdStation --intensity 100
nohup python top.py --mode ISL --intensity 100

