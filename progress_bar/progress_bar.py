def percentage_to_bar(percentage: float):
    int_per = int(percentage)
    res = int_per % 5
    done = int(((int_per - res) + 1) / 5)
    to_do = int((100 - int_per + res) / 5)
    bar = "|" + int(done) * "█" + int(to_do) * "_" + "|" + \
          "{0:3.0f}%".format(percentage)
    return bar