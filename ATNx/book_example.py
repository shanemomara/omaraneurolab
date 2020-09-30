"""This is an example from Information Theory and Analysis."""

from bayes_atnx import bayes_stats

def main():
    res = bayes_stats(
        10,
        3,
        30,
        1,
        bayes_les_prob=0.1,
        srate=100
    )
    return res

if __name__ == '__main__':
    print(main())