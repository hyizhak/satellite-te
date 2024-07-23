for intensity in 25; do
    for mode in ISL GrdStation; do
        python top.py --intensity $intensity --mode $mode --POP-ratio 128
    done
done