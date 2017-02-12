import sys
from collections import defaultdict, Counter
LATEX=True

# prints grouped ghkm rules by LHS

def is_ordered(rhs):
    rhs = [int(x[1:]) for x in rhs.split() if x[0] == "x"]
    positive = [(b-a)>0 for a,b in zip(rhs,rhs[1:])]
    return all(positive)

rc = Counter()
rules = defaultdict(Counter)
for line in sys.stdin:
    rule, s1,s2,c = line.strip().split("\t")
    LHS, RHS = rule.split(" -> ")
    if "." in RHS: RHS = RHS.replace(".",",/.")
    else: RHS = RHS.replace(",",",/.")
    rules[LHS][RHS] += int(float(c))
    if not is_ordered(RHS):
        rc[LHS] += int(float(c))

for LHS, cnt in rc.most_common():
    RHSs = rules[LHS]
    if not(all([is_ordered(r) for r,c in RHSs.most_common()])):
        if not LATEX:
            print LHS
            print "\t"," ".join(map(str,RHSs))
        if LATEX:
            print LHS,"&",
            for r,c in RHSs.most_common()[:5]:
                if not is_ordered(r): r = "\\textbf{%s}" % r
                print "(%s)" % c,r,"$\;\;$",
            print "\\\\"