# -*- coding: utf-8 -*-

import sys, re
def run(line):
    output = []
    words = line.strip().split("\t")
    output_line = words[3]+" $$ "+words[4]+" $$ "+words[0]
    return output_line


if __name__ == "__main__":

    infilename = sys.argv[1]
    infile = open(infilename,'r')
    # infile = ["1	702876	702977	Amrozi accused his brother, whom he called \"the witness\", of deliberately distorting his evidence.	Referring to him as only \"the witness\", Amrozi accused his brother of deliberately distorting his evidence.",
				# "0	2108705	2108831	Yucaipa owned Dominick's before selling the chain to Safeway in 1998 for $2.5 billion.	Yucaipa bought Dominick's in 1995 for $693 million and sold it to Safeway for $1.8 billion in 1998.",
				# "1	1330381	1330521	They had published an advertisement on the Internet on June 10, offering the cargo for sale, he added.	On June 10, the ship's owners had published an advertisement on the Internet, offering the explosives for sale.",
				# "0	3344667	3344648	Around 0335 GMT, Tab shares were up 19 cents, or 4.4%, at A$4.56, having earlier set a record high of A$4.57.	Tab shares jumped 20 cents, or 4.6%, to set a record closing high at A$4.57."]
    for line in infile:
        print run(line)