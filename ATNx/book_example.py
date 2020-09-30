"""This is an example from Information Theory and Analysis."""

from bayes_atnx import bayes_stats

def main():
    res = bayes_stats(
        30,
        1,
        10,
        3,
        bayes_les_prob=1.0,
        srate=100
    )
    return res

if __name__ == '__main__':
    print(main())