for size in 176 500 528 1500; do
    for mode in GrdStation ISL; do
        nohup python lp.py --mode $mode --size $size
        # nohup python lp.py --intensity $intensity --mode $mode --top-percentage 0.1
    done
done

nohup python lp.py --mode GrdStation --intensity 100
nohup python lp.py --mode ISL --intensity 100

