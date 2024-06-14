for radius in .001  .0003 .0001 
do
    for tradeoff in .3 1 3 10 
    do 
        for align in  1 3 .3 10
        do
            sbatch run.sh $radius $tradeoff $align
            # bash run.sh $radius $tradeoff $align
        done
    done
done