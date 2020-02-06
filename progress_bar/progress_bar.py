def percentage_to_bar(percentage: float):
    int_per = int(percentage)
    res = int_per % 5
    done = (int_per - res)
    to_do = 100 - done
    done = int(done / 5)
    to_do = int(to_do / 5)
    bar = "|" + int(done) * "█" + int(to_do) * "_" + "|" + "{0:3.0f}%".format(percentage)
    return bar
